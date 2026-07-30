"""
streamlit_app.py
-----------------
MediLeaf AI - Streamlit entry point.
Orchestrates UI and session state; all logic (PDF parsing, embeddings,
retrieval, prompting, Gemini call, memory, emergency detection, logging)
lives in the app/ and utils/ packages.
"""

from __future__ import annotations

import time

import streamlit as st

from app.config import QUICK_QUESTIONS, settings
from app.emergency import detect_emergency
from app.llm import LLMError, StructuredAnswer, answer_question, generate_summary
from app.memory import ConversationMemory, SearchHistory
from app.parser import PDFParsingError, chunk_pages, extract_pages, full_text
from app.retriever import (
    build_vector_store,
    has_sufficient_context,
    overall_confidence,
    retrieve,
)
from utils.helpers import new_session_id, truncate
from utils.logger import log_answer, log_error, log_feedback, log_question, log_retrieval
# import os
# from dotenv import load_dotenv

# load_dotenv()

# st.write("API KEY:", os.getenv("GOOGLE_API_KEY"))
# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="MediLeaf AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Custom CSS (Glassmorphism UI)
# ---------------------------
st.markdown("""
<style>

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0F1117;
    color: #F5F7FA;
}

/* Glass Card */
.glass-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid #2A2E38;
    border-radius: 16px;
    padding: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    margin-bottom: 16px;
    transition: 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 30px rgba(0,0,0,0.4);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #2E8B57, #3CB371);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 10px 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 12px #3CB371;
}

/* Chat bubbles */
.user-msg {
    background: #2E8B57;
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 8px;
    text-align: right;
}

.bot-msg {
    background: rgba(255,255,255,0.06);
    padding: 12px;
    border-radius: 12px;
    margin-bottom: 8px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #171A22;
}

/* Input */
input {
    border-radius: 999px !important;
    border: 1px solid #2E8B57 !important;
}

/* Upload */
.upload-box {
    border: 2px dashed #2E8B57;
    border-radius: 16px;
    padding: 30px;
    text-align: center;
}

/* Chips */
.chip {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
    margin: 4px;
    cursor: pointer;
    transition: 0.3s;
}

.chip:hover {
    background: #3CB371;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# Session State Initialisation
# ===========================
if "session_id" not in st.session_state:
    st.session_state.session_id = new_session_id()
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "leaflet_names" not in st.session_state:
    st.session_state.leaflet_names = []
if "summary" not in st.session_state:
    st.session_state.summary = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "search_history" not in st.session_state:
    st.session_state.search_history = SearchHistory()
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "feedback" not in st.session_state:
    st.session_state.feedback = {}


# ===========================
# Core Actions
# ===========================
def handle_upload(uploaded_files: list) -> None:
    """Parse, chunk, embed and index uploaded leaflet PDFs."""
    all_chunks = []
    all_pages_text = []
    names = []

    with st.spinner("Reading and understanding your leaflet(s)..."):
        for uploaded_file in uploaded_files:
            try:
                pages = extract_pages(uploaded_file, uploaded_file.name)
                chunks = chunk_pages(pages, uploaded_file.name)
            except PDFParsingError as exc:
                st.error(f"⚠️ {exc}")
                log_error(st.session_state.session_id, "pdf_parsing", str(exc))
                continue
            except Exception as exc:
                st.error(f"⚠️ Unexpected error reading '{uploaded_file.name}': {exc}")
                log_error(st.session_state.session_id, "pdf_parsing", str(exc))
                continue

            all_chunks.extend(chunks)
            all_pages_text.append(full_text(pages))
            names.append(uploaded_file.name)

        if not all_chunks:
            st.error("No leaflet could be processed. Please check the file(s) and try again.")
            return

        try:
            store = build_vector_store(all_chunks, st.session_state.session_id)
        except Exception as exc:
            st.error(f"⚠️ Could not build the search index: {exc}")
            log_error(st.session_state.session_id, "vector_store", str(exc))
            return

        st.session_state.vector_store = store
        st.session_state.leaflet_names = names
        st.session_state.memory.clear()
        st.session_state.search_history.clear()
        st.session_state.chat_log = []
        st.session_state.feedback = {}

        combined_text = "\n\n".join(all_pages_text)
        try:
            st.session_state.summary = generate_summary(combined_text)
        except LLMError as exc:
            st.session_state.summary = None
            st.warning(f"Leaflet indexed, but the automatic summary could not be generated: {exc}")
            log_error(st.session_state.session_id, "summary", str(exc))

    st.success(f"✅ {len(names)} leaflet(s) indexed: {', '.join(names)}")


def handle_question(question: str) -> None:
    """Run the full RAG pipeline for one user question and store the result."""
    question = question.strip()
    if not question:
        return

    if st.session_state.vector_store is None:
        st.warning("Please upload a medicine leaflet PDF first.")
        return

    session_id = st.session_state.session_id
    log_question(session_id, question)
    emergency_terms = detect_emergency(question)

    t0 = time.perf_counter()
    try:
        chunks = retrieve(st.session_state.vector_store, question)
        confidence = overall_confidence(chunks)
        sufficient = has_sufficient_context(chunks)
        retrieval_time = time.perf_counter() - t0
        log_retrieval(session_id, question, len(chunks), confidence, retrieval_time)

        answer = answer_question(
            question=question,
            chunks=chunks if sufficient else [],
            memory_context=st.session_state.memory.as_prompt_context(),
            overall_confidence=confidence,
        )
        log_answer(session_id, question, answer.execution_time_seconds, answer.insufficient_information)

    except LLMError as exc:
        st.error(f"⚠️ The AI service could not process this question: {exc}")
        log_error(session_id, "llm", str(exc))
        return
    except Exception as exc:
        st.error(f"⚠️ Something went wrong answering that question: {exc}")
        log_error(session_id, "pipeline", str(exc))
        return

    st.session_state.memory.add_turn(question, answer.answer or settings.insufficient_info_message)
    st.session_state.search_history.add(question)
    st.session_state.chat_log.append(
        {"question": question, "answer": answer, "emergency": emergency_terms}
    )


# ===========================
# Sidebar
# ===========================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🌿 MediLeaf AI")

        st.markdown("### 📤 Upload PDF")
        uploaded_files = st.file_uploader(
            "Upload one or more medicine leaflet PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files and st.button("Process leaflet(s)", type="primary", use_container_width=True):
            handle_upload(uploaded_files)

        if st.session_state.leaflet_names:
            st.caption("Currently loaded: " + ", ".join(st.session_state.leaflet_names))

        st.divider()

        st.markdown("### 📊 Medicine Snapshot")
        if st.session_state.summary:
            render_summary_sidebar(st.session_state.summary)
        else:
            st.caption("Upload a leaflet to see an automatic summary here.")

        st.divider()

        st.markdown("### 🕘 Search History")
        history = st.session_state.search_history.all()
        if not history:
            st.caption("Your previous questions will appear here.")
        else:
            for i, past_question in enumerate(reversed(history[-20:])):
                if st.button(truncate(past_question, 42), key=f"hist_{i}_{past_question}"):
                    st.session_state.pending_question = past_question

        st.divider()

        with st.expander("⚙️ Settings"):
            st.caption(f"LLM model: `{settings.llm_model_name}`")
            st.caption(f"Temperature: `{settings.llm_temperature}`")
            st.caption(f"Embedding model: `{settings.embedding_model_name}`")
            st.caption(f"Retrieval depth (top-k): `{settings.top_k}`")
            st.caption(f"Memory window: last `{settings.memory_window}` turns")
            if st.button("🗑️ Clear session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

        with st.expander("ℹ️ About"):
            st.write(
                f"**{settings.app_name}** helps you understand medicine leaflets in "
                "simple language. It answers strictly from the leaflet you upload, "
                "using Retrieval-Augmented Generation so every answer can be traced "
                "back to a specific page."
            )
            st.caption(settings.disclaimer)


def render_summary_sidebar(summary) -> None:
    if summary.medicine_name:
        st.markdown(f"**{summary.medicine_name}**")
    if summary.drug_class:
        st.caption(summary.drug_class)
    fields = [
        ("Uses", summary.uses),
        ("Dosage", summary.dosage_instructions),
        ("Common side effects", summary.common_side_effects),
        ("Warnings", summary.warnings),
    ]
    for label, value in fields:
        if value:
            with st.expander(label):
                st.write(value)


# ===========================
# Main Content
# ===========================
def render_header() -> None:
    st.markdown(
        f"""
        <div class="ml-header">
            <h1>🌿 {settings.app_name}</h1>
            <p>{settings.app_tagline}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_full_summary(summary) -> None:
    st.markdown("### 📋 Automatic Leaflet Summary")
    left, right = st.columns(2)
    left_fields = [
        ("Medicine Name", summary.medicine_name),
        ("Drug Class", summary.drug_class),
        ("Uses", summary.uses),
        ("Who Should Not Use It", summary.who_should_not_use_it),
        ("Pregnancy", summary.pregnancy),
        ("Breastfeeding", summary.breastfeeding),
        ("Children", summary.children),
        ("Elderly", summary.elderly),
    ]
    right_fields = [
        ("Dosage Instructions", summary.dosage_instructions),
        ("Missed Dose", summary.missed_dose),
        ("Overdose", summary.overdose),
        ("Storage", summary.storage),
        ("Common Side Effects", summary.common_side_effects),
        ("Serious Side Effects", summary.serious_side_effects),
        ("Warnings", summary.warnings),
        ("When To Contact A Doctor", summary.when_to_contact_a_doctor),
    ]
    for col, fields in ((left, left_fields), (right, right_fields)):
        with col:
            for label, value in fields:
                if value:
                    st.markdown(
                        f'<div class="glass-card"><h4>{label}</h4><p>{value}</p></div>',
                        unsafe_allow_html=True,
                    )


def render_quick_questions() -> None:
    st.markdown("##### 💡 Quick questions")
    cols = st.columns(4)
    for i, q in enumerate(QUICK_QUESTIONS):
        if cols[i % 4].button(q, key=f"quick_{i}"):
            st.session_state.pending_question = q


def render_answer_card(entry: dict, index: int) -> None:
    question = entry["question"]
    answer: StructuredAnswer = entry["answer"]
    emergency_terms = entry["emergency"]

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if emergency_terms:
            from app.emergency import EMERGENCY_MESSAGE
            st.markdown(
                f'<div style="background:#FF6B6B;padding:20px;border-radius:16px;color:white;'
                f'box-shadow:0 4px 20px rgba(0,0,0,0.3);">🚨 {EMERGENCY_MESSAGE}</div>',
                unsafe_allow_html=True,
            )

        if answer.insufficient_information:
            st.markdown(
                f'<div class="glass-card"><h4>Answer</h4><p>{answer.answer}</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="glass-card"><h4>Answer</h4><p>{answer.answer}</p></div>',
                unsafe_allow_html=True,
            )
            if answer.important_information:
                st.markdown(
                    f'<div class="glass-card"><h4>Important Information</h4>'
                    f"<p>{answer.important_information}</p></div>",
                    unsafe_allow_html=True,
                )
            if answer.warnings:
                st.markdown(
                    f'<div class="glass-card" style="border-left: 4px solid #FF6B6B;"><h4>Warnings</h4>'
                    f"<p>{answer.warnings}</p></div>",
                    unsafe_allow_html=True,
                )
            if answer.practical_advice:
                st.markdown(
                    f'<div class="glass-card"><h4>Practical Advice</h4>'
                    f"<p>{answer.practical_advice}</p></div>",
                    unsafe_allow_html=True,
                )
            if answer.sources:
                source_lines = "<br>".join(
                    f"Page {s['page']} — {s['section']}" for s in answer.sources
                )
                st.markdown(
                    f'<div class="glass-card"><h4>Source</h4>'
                    f"<p>{source_lines}</p></div>",
                    unsafe_allow_html=True,
                )
            if answer.explanation:
                st.caption(f"🔍 Why this answer: {answer.explanation}")

        st.markdown(f"**Understanding Confidence: {answer.confidence:.0f}%**")
        st.progress(answer.confidence / 100)
        st.markdown(f'<small>{answer.disclaimer}</small>', unsafe_allow_html=True)

        fb_col1, fb_col2, _ = st.columns([1, 1, 6])
        current_feedback = st.session_state.feedback.get(index)
        if fb_col1.button("👍 Helpful", key=f"fb_up_{index}", disabled=current_feedback is not None):
            st.session_state.feedback[index] = "up"
            log_feedback(st.session_state.session_id, question, True)
            st.rerun()
        if fb_col2.button("👎 Not Helpful", key=f"fb_down_{index}", disabled=current_feedback is not None):
            st.session_state.feedback[index] = "down"
            log_feedback(st.session_state.session_id, question, False)
            st.rerun()
        if current_feedback:
            st.caption("Thanks for your feedback!" if current_feedback == "up" else "Thanks — we'll use this to improve.")


# ===========================
# Page Layout
# ===========================
def main() -> None:
    problems = settings.validate()
    render_sidebar()
    render_header()

    if problems:
        for problem in problems:
            st.error(f"⚠️ Configuration issue: {problem}")

    if st.session_state.vector_store is None:
        st.info("👈 Upload a medicine leaflet PDF from the sidebar to get started.")
        return

    if st.session_state.summary:
        with st.expander("📋 View automatic leaflet summary", expanded=len(st.session_state.chat_log) == 0):
            render_full_summary(st.session_state.summary)

    render_quick_questions()
    st.divider()

    for i, entry in enumerate(st.session_state.chat_log):
        render_answer_card(entry, i)

    typed_question = st.chat_input("Ask a question about this leaflet...")

    question_to_run = None
    if st.session_state.pending_question:
        question_to_run = st.session_state.pending_question
        st.session_state.pending_question = None
    elif typed_question:
        question_to_run = typed_question

    if question_to_run:
        handle_question(question_to_run)
        st.rerun()


if __name__ == "__main__":
    main()

