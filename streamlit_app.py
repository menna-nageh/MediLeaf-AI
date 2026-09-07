"""
streamlit_app.py
----------------
MediLeaf AI - Streamlit entry point.

Responsibilities:
- Premium Streamlit UI
- Session state management
- PDF upload orchestration
- RAG pipeline orchestration
- Conversation history
- Retrieval / grounding visibility
- Citation rendering
- Feedback collection

Core RAG logic remains inside app/.
"""

from __future__ import annotations

import html
import time
from typing import Any

import streamlit as st

from app.config import QUICK_QUESTIONS, settings
from app.emergency import EMERGENCY_MESSAGE, detect_emergency
from app.llm import (
    LLMError,
    StructuredAnswer,
    answer_question,
    generate_summary,
)
from app.memory import ConversationMemory, SearchHistory
from app.monitoring import record_feedback, record_request
from app.parser import (
    PDFParsingError,
    chunk_pages,
    extract_pages,
    full_text,
)
from app.retriever import (
    build_vector_store,
    has_sufficient_context,
    overall_confidence,
    retrieve,
)
from utils.helpers import new_session_id, truncate
from utils.logger import (
    log_answer,
    log_error,
    log_feedback,
    log_question,
    log_retrieval,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="MediLeaf AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown(
    """
<style>

:root {
    --ml-bg: #080c0a;
    --ml-bg-soft: #0c120f;
    --ml-surface: rgba(255,255,255,0.035);
    --ml-surface-strong: rgba(255,255,255,0.055);
    --ml-border: rgba(255,255,255,0.075);
    --ml-border-green: rgba(94,229,138,0.18);
    --ml-text: #f3f7f4;
    --ml-text-soft: #d3ddd7;
    --ml-muted: #91a098;
    --ml-green: #5ee58a;
}

/* -------------------------------------------------------------------------
   GLOBAL
   ------------------------------------------------------------------------- */

html,
body,
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at 8% 5%,
            rgba(46,139,87,0.13),
            transparent 28%
        ),
        radial-gradient(
            circle at 92% 75%,
            rgba(94,229,138,0.055),
            transparent 26%
        ),
        var(--ml-bg);
    color: var(--ml-text);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
    padding-bottom: 5rem;
}

/* -------------------------------------------------------------------------
   SIDEBAR
   ------------------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0a100d 0%,
            #0d1511 100%
        );
    border-right: 1px solid var(--ml-border);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1.25rem;
}

.ml-sidebar-brand {
    padding: 4px 0 20px;
}

.ml-sidebar-logo {
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.04em;
}

.ml-sidebar-subtitle {
    margin-top: 5px;
    color: var(--ml-muted);
    font-size: 0.79rem;
    line-height: 1.5;
}

/* -------------------------------------------------------------------------
   HEADER
   ------------------------------------------------------------------------- */

.ml-header {
    position: relative;
    overflow: hidden;
    padding: 38px 42px;
    margin-bottom: 22px;
    border: 1px solid var(--ml-border-green);
    border-radius: 26px;
    background:
        linear-gradient(
            135deg,
            rgba(46,139,87,0.17),
            rgba(255,255,255,0.025)
        );
    box-shadow:
        0 25px 80px rgba(0,0,0,0.25);
}

.ml-header::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    right: -130px;
    top: -170px;
    border-radius: 50%;
    background: rgba(94,229,138,0.08);
}

.ml-header::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: 120px;
    bottom: -130px;
    border-radius: 50%;
    background: rgba(46,139,87,0.07);
    filter: blur(10px);
}

.ml-header-content {
    position: relative;
    z-index: 2;
}

.ml-header-eyebrow {
    display: inline-flex;
    padding: 6px 11px;
    margin-bottom: 13px;
    border-radius: 999px;
    color: #baf6cc;
    background: rgba(94,229,138,0.08);
    border: 1px solid rgba(94,229,138,0.15);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}

.ml-header h1 {
    margin: 0;
    font-size: 2.45rem;
    line-height: 1.1;
    font-weight: 800;
    letter-spacing: -0.055em;
}

.ml-header p {
    margin: 10px 0 0;
    max-width: 720px;
    color: var(--ml-muted);
    font-size: 1rem;
    line-height: 1.65;
}

/* -------------------------------------------------------------------------
   STATUS
   ------------------------------------------------------------------------- */

.ml-status-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 22px;
}

.ml-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,0.035);
    border: 1px solid var(--ml-border);
    color: #cbd6d0;
    font-size: 0.74rem;
    font-weight: 650;
}

.ml-status-active {
    color: #baf6cc;
    background: rgba(94,229,138,0.075);
    border-color: rgba(94,229,138,0.15);
}

/* -------------------------------------------------------------------------
   CARDS
   ------------------------------------------------------------------------- */

.ml-card {
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.052),
            rgba(255,255,255,0.018)
        );
    border: 1px solid var(--ml-border);
    border-radius: 18px;
    padding: 20px 22px;
    margin-bottom: 14px;
    box-shadow:
        0 12px 38px rgba(0,0,0,0.14);
}

.ml-card-title {
    color: var(--ml-text);
    font-size: 1.02rem;
    font-weight: 750;
    margin-bottom: 8px;
}

.ml-card-text {
    color: var(--ml-text-soft);
    line-height: 1.75;
}

/* -------------------------------------------------------------------------
   EMPTY STATE
   ------------------------------------------------------------------------- */

.ml-empty {
    text-align: center;
    padding: 80px 30px;
    border: 1px dashed rgba(94,229,138,0.22);
    border-radius: 26px;
    background:
        linear-gradient(
            145deg,
            rgba(46,139,87,0.08),
            rgba(255,255,255,0.018)
        );
}

.ml-empty-icon {
    width: 72px;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 18px;
    border-radius: 22px;
    background: rgba(94,229,138,0.08);
    border: 1px solid rgba(94,229,138,0.13);
    font-size: 2.2rem;
}

.ml-empty h2 {
    margin: 0 0 9px;
    font-size: 1.45rem;
}

.ml-empty p {
    max-width: 600px;
    margin: auto;
    color: var(--ml-muted);
    line-height: 1.65;
}

.ml-empty-flow {
    margin-top: 24px;
    color: #8fa097;
    font-size: 0.82rem;
    font-weight: 650;
}

/* -------------------------------------------------------------------------
   DOCUMENTS
   ------------------------------------------------------------------------- */

.ml-document {
    padding: 12px 13px;
    margin-bottom: 8px;
    border-radius: 13px;
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.06);
    color: #dce5df;
    font-size: 0.82rem;
    overflow-wrap: anywhere;
}

/* -------------------------------------------------------------------------
   SUMMARY
   ------------------------------------------------------------------------- */

.ml-summary-card {
    min-height: 105px;
    padding: 17px 18px;
    border-radius: 16px;
    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.045),
            rgba(255,255,255,0.018)
        );
    border: 1px solid var(--ml-border);
    margin-bottom: 12px;
}

.ml-summary-label {
    color: var(--ml-muted);
    font-size: 0.70rem;
    font-weight: 750;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.ml-summary-value {
    color: var(--ml-text-soft);
    font-size: 0.91rem;
    line-height: 1.65;
}

/* -------------------------------------------------------------------------
   SOURCES
   ------------------------------------------------------------------------- */

.ml-source {
    padding: 14px 15px;
    margin-top: 9px;
    border-radius: 14px;
    background: rgba(255,255,255,0.032);
    border: 1px solid rgba(255,255,255,0.065);
}

.ml-source-title {
    color: #edf4ef;
    font-size: 0.88rem;
    font-weight: 700;
}

.ml-source-meta {
    margin-top: 6px;
    color: var(--ml-muted);
    font-size: 0.74rem;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

.ml-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    margin: 7px 5px 0 0;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
}

.ml-badge-green {
    color: #baf6cc;
    background: rgba(94,229,138,0.09);
    border: 1px solid rgba(94,229,138,0.15);
}

.ml-badge-muted {
    color: #b7c3bc;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.07);
}

/* -------------------------------------------------------------------------
   EMERGENCY
   ------------------------------------------------------------------------- */

.ml-emergency {
    padding: 18px 20px;
    margin-bottom: 18px;
    border-radius: 16px;
    background:
        linear-gradient(
            135deg,
            rgba(255,107,107,0.16),
            rgba(255,107,107,0.045)
        );
    border: 1px solid rgba(255,107,107,0.32);
    color: #ffe7e7;
    line-height: 1.65;
}

/* -------------------------------------------------------------------------
   BUTTONS
   ------------------------------------------------------------------------- */

div[data-testid="stButton"] > button {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.035);
    color: #dce6e0;
    transition:
        transform 0.18s ease,
        border-color 0.18s ease,
        background 0.18s ease;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: rgba(94,229,138,0.30);
    background: rgba(94,229,138,0.075);
    color: white;
}

button[kind="primary"] {
    background:
        linear-gradient(
            135deg,
            #2e8b57,
            #42b96f
        ) !important;
    border: none !important;
    box-shadow:
        0 8px 28px rgba(46,139,87,0.22);
}

/* -------------------------------------------------------------------------
   CHAT
   ------------------------------------------------------------------------- */

[data-testid="stChatMessage"] {
    border-radius: 18px;
    margin-bottom: 8px;
}

[data-testid="stChatMessageContent"] {
    line-height: 1.75;
}

/* -------------------------------------------------------------------------
   INPUT
   ------------------------------------------------------------------------- */

[data-testid="stChatInput"] {
    border-color: rgba(94,229,138,0.20);
}

/* -------------------------------------------------------------------------
   METRICS
   ------------------------------------------------------------------------- */

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--ml-border);
    border-radius: 15px;
    padding: 12px 14px;
}

/* -------------------------------------------------------------------------
   EXPANDERS
   ------------------------------------------------------------------------- */

[data-testid="stExpander"] {
    border-color: var(--ml-border);
    border-radius: 14px;
}

/* -------------------------------------------------------------------------
   FOOTER
   ------------------------------------------------------------------------- */

.ml-footer {
    margin-top: 45px;
    padding-top: 18px;
    border-top: 1px solid rgba(255,255,255,0.06);
    text-align: center;
    color: #68756e;
    font-size: 0.72rem;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================================
# SESSION STATE
# ============================================================================


def initialize_session_state() -> None:
    """Initialize Streamlit session state."""

    defaults = {
        "session_id": new_session_id(),
        "vector_store": None,
        "leaflet_names": [],
        "summary": None,
        "memory": ConversationMemory(),
        "search_history": SearchHistory(),
        "chat_log": [],
        "pending_question": None,
        "feedback": {},
        "index_stats": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================================
# HELPERS
# ============================================================================


def safe_text(value: Any) -> str:
    """Escape dynamic text before putting it inside HTML."""

    if value is None:
        return ""

    return html.escape(
        str(value)
    )


def get_value(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """Read a value from either a dict or an object."""

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def reset_session() -> None:
    """Reset the current Streamlit session."""

    for key in list(
        st.session_state.keys()
    ):
        del st.session_state[key]

    st.rerun()


# ============================================================================
# UPLOAD
# ============================================================================


def handle_upload(
    uploaded_files: list,
) -> None:
    """
    Parse, chunk, embed and index uploaded leaflet PDFs.
    """

    all_chunks = []
    all_pages_text = []
    names = []

    progress = st.progress(0)
    status = st.empty()

    total = len(
        uploaded_files
    )

    with st.spinner(
        "Processing your medicine leaflet..."
    ):

        for index, uploaded_file in enumerate(
            uploaded_files
        ):

            status.info(
                f"Reading **{uploaded_file.name}** "
                f"({index + 1}/{total})"
            )

            try:

                pages = extract_pages(
                    uploaded_file,
                    uploaded_file.name,
                )

                chunks = chunk_pages(
                    pages,
                    uploaded_file.name,
                )

            except PDFParsingError as exc:

                st.error(
                    f"Could not process "
                    f"`{uploaded_file.name}`: {exc}"
                )

                log_error(
                    st.session_state.session_id,
                    "pdf_parsing",
                    str(exc),
                )

                continue

            except Exception as exc:

                st.error(
                    f"Unexpected error reading "
                    f"`{uploaded_file.name}`: {exc}"
                )

                log_error(
                    st.session_state.session_id,
                    "pdf_parsing",
                    str(exc),
                )

                continue

            all_chunks.extend(
                chunks
            )

            all_pages_text.append(
                full_text(pages)
            )

            names.append(
                uploaded_file.name
            )

            progress.progress(
                min(
                    (index + 1)
                    / max(total, 1),
                    1.0,
                )
            )

        status.empty()

        if not all_chunks:

            progress.empty()

            st.error(
                "No leaflet could be processed. "
                "Please check your PDF files."
            )

            return

        # --------------------------------------------------------------------
        # Build vector index
        # --------------------------------------------------------------------

        try:

            with st.spinner(
                "Building the hybrid search index..."
            ):

                store = build_vector_store(
                    all_chunks,
                    st.session_state.session_id,
                )

        except Exception as exc:

            progress.empty()

            st.error(
                f"Could not build the search index: {exc}"
            )

            log_error(
                st.session_state.session_id,
                "vector_store",
                str(exc),
            )

            return

        # --------------------------------------------------------------------
        # Session state
        # --------------------------------------------------------------------

        st.session_state.vector_store = store
        st.session_state.leaflet_names = names

        st.session_state.memory.clear()
        st.session_state.search_history.clear()
        st.session_state.chat_log = []
        st.session_state.feedback = {}

        st.session_state.index_stats = {
            "documents": len(names),
            "chunks": len(all_chunks),
        }

        # --------------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------------

        combined_text = "\n\n".join(
            all_pages_text
        )

        try:

            with st.spinner(
                "Creating your leaflet overview..."
            ):

                summary = generate_summary(
                    combined_text
                )

                st.session_state.summary = (
                    summary
                )

        except LLMError as exc:

            st.session_state.summary = None

            st.warning(
                "The leaflet was indexed successfully, "
                "but the automatic summary could not be generated."
            )

            log_error(
                st.session_state.session_id,
                "summary",
                str(exc),
            )

        progress.progress(1.0)

    st.success(
        f"✓ {len(names)} leaflet(s) indexed successfully."
    )


# ============================================================================
# QUESTION HANDLER
# ============================================================================


def handle_question(
    question: str,
) -> None:
    """
    Execute the full RAG pipeline for one user question.
    """

    question = question.strip()

    if not question:
        return

    if st.session_state.vector_store is None:

        st.warning(
            "Please upload a medicine leaflet PDF first."
        )

        return

    session_id = (
        st.session_state.session_id
    )

    log_question(
        session_id,
        question,
    )

    emergency_terms = detect_emergency(
        question
    )

    retrieval_start = time.perf_counter()

    try:

        # --------------------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------------------

        chunks = retrieve(
            st.session_state.vector_store,
            question,
        )

        retrieval_confidence = (
            overall_confidence(
                chunks
            )
        )

        sufficient = has_sufficient_context(
            chunks
        )

        retrieval_time = (
            time.perf_counter()
            - retrieval_start
        )

        log_retrieval(
            session_id,
            question,
            len(chunks),
            retrieval_confidence,
            retrieval_time,
            retrieval_mode=settings.retrieval_mode,
        )

        # --------------------------------------------------------------------
        # Generation
        # --------------------------------------------------------------------

        answer = answer_question(
            question=question,
            chunks=(
                chunks
                if sufficient
                else []
            ),
            memory_context=(
                st.session_state.memory
                .as_prompt_context()
            ),
            overall_confidence=(
                retrieval_confidence
            ),
        )

        log_answer(
            session_id,
            question,
            answer.execution_time_seconds,
            answer.insufficient_information,
            confidence=answer.confidence,
            grounded=answer.grounded,
            grounding_score=answer.grounding_score,
            retrieval_mode=settings.retrieval_mode,
        )
        record_request(
            {
                "session_id": session_id,
                "question": question,
                "answer": answer.answer,
                "latency_seconds": answer.execution_time_seconds,
                "retrieval_mode": settings.retrieval_mode,
                "confidence": answer.confidence,
                "grounded": answer.grounded,
                "grounding_score": answer.grounding_score,
                "insufficient_information": answer.insufficient_information,
            }
        )

    except LLMError as exc:

        st.error(
            "The AI service could not process "
            f"this question: {exc}"
        )

        log_error(
            session_id,
            "llm",
            str(exc),
        )

        return

    except Exception as exc:

        st.error(
            "Something went wrong while answering: "
            f"{exc}"
        )

        log_error(
            session_id,
            "pipeline",
            str(exc),
        )

        return

    # ------------------------------------------------------------------------
    # Save conversation state
    # ------------------------------------------------------------------------

    memory_answer = (
        answer.answer
        or settings.insufficient_info_message
    )

    st.session_state.memory.add_turn(
        question,
        memory_answer,
    )

    st.session_state.search_history.add(
        question
    )

    st.session_state.chat_log.append(
        {
            "question": question,
            "answer": answer,
            "emergency": emergency_terms,
        }
    )


# ============================================================================
# SIDEBAR
# ============================================================================


def render_sidebar() -> None:

    with st.sidebar:

        # --------------------------------------------------------------------
        # Brand
        # --------------------------------------------------------------------

        st.markdown(
            """
            <div class="ml-sidebar-brand">
                <div class="ml-sidebar-logo">
                    🌿 MediLeaf AI
                </div>

                <div class="ml-sidebar-subtitle">
                    Grounded medicine leaflet assistant
                    powered by retrieval-augmented generation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------------------
        # Upload
        # --------------------------------------------------------------------

        st.markdown(
            "### 📄 Medicine leaflet"
        )

        uploaded_files = st.file_uploader(
            "Upload one or more medicine leaflet PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:

            st.caption(
                f"{len(uploaded_files)} PDF file(s) selected"
            )

            if st.button(
                "Process leaflet(s)",
                type="primary",
                use_container_width=True,
            ):

                handle_upload(
                    uploaded_files
                )

        else:

            st.caption(
                "Upload a PDF leaflet to start."
            )

        # --------------------------------------------------------------------
        # Loaded documents
        # --------------------------------------------------------------------

        if st.session_state.leaflet_names:

            st.divider()

            st.markdown(
                "### 📚 Loaded documents"
            )

            for name in (
                st.session_state.leaflet_names
            ):

                st.markdown(
                    f"""
                    <div class="ml-document">
                        📄 {safe_text(name)}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # --------------------------------------------------------------------
        # Snapshot
        # --------------------------------------------------------------------

        st.divider()

        st.markdown(
            "### 💊 Medicine snapshot"
        )

        if st.session_state.summary:

            render_summary_sidebar(
                st.session_state.summary
            )

        else:

            st.caption(
                "The automatic snapshot will appear "
                "after indexing a leaflet."
            )

        # --------------------------------------------------------------------
        # Recent questions
        # --------------------------------------------------------------------

        st.divider()

        st.markdown(
            "### 🕘 Recent questions"
        )

        history = (
            st.session_state.search_history.all()
        )

        if not history:

            st.caption(
                "Your recent questions will appear here."
            )

        else:

            recent = list(
                reversed(
                    history[-10:]
                )
            )

            for index, past_question in enumerate(
                recent
            ):

                if st.button(
                    truncate(
                        past_question,
                        38,
                    ),
                    key=(
                        f"history_{index}_"
                        f"{past_question}"
                    ),
                    use_container_width=True,
                ):

                    st.session_state.pending_question = (
                        past_question
                    )

        # --------------------------------------------------------------------
        # System settings
        # --------------------------------------------------------------------

        st.divider()

        with st.expander(
            "⚙️ System",
            expanded=False,
        ):

            st.caption(
                f"LLM: `{settings.llm_model_name}`"
            )

            st.caption(
                "Embeddings: "
                f"`{settings.embedding_model_name}`"
            )

            st.caption(
                "Retrieval: "
                f"`{settings.retrieval_mode}`"
            )

            st.caption(
                f"Top-k: `{settings.top_k}`"
            )

            st.caption(
                "Reranker: "
                f"`{settings.reranker_model_name}`"
            )

            st.caption(
                "Reranker status: "
                f"`{'ON' if settings.reranker_enabled else 'OFF'}`"
            )

            st.caption(
                "Memory: "
                f"`{settings.memory_window}` turns"
            )

        # --------------------------------------------------------------------
        # Clear
        # --------------------------------------------------------------------

        if st.button(
            "🗑️ Clear session",
            use_container_width=True,
        ):

            reset_session()

        # --------------------------------------------------------------------
        # About
        # --------------------------------------------------------------------

        with st.expander(
            "ℹ️ About MediLeaf",
        ):

            st.write(
                "MediLeaf AI answers questions using "
                "the uploaded medicine leaflet as its "
                "primary knowledge source."
            )

            st.caption(
                "RAG • BM25 • Vector Search • Reranking "
                "• Citations • Grounding"
            )

            st.caption(
                settings.disclaimer
            )


# ============================================================================
# SUMMARY SIDEBAR
# ============================================================================


def render_summary_sidebar(
    summary: Any,
) -> None:
    """Render a compact medicine snapshot."""

    fields = [
        (
            "Medicine",
            "medicine_name",
        ),
        (
            "Drug class",
            "drug_class",
        ),
        (
            "Uses",
            "uses",
        ),
        (
            "Dosage",
            "dosage_instructions",
        ),
        (
            "Side effects",
            "common_side_effects",
        ),
        (
            "Warnings",
            "warnings",
        ),
    ]

    for label, attribute in fields:

        value = get_value(
            summary,
            attribute,
            "",
        )

        if value:

            with st.expander(
                label
            ):

                st.write(
                    value
                )


# ============================================================================
# HEADER
# ============================================================================


def render_header() -> None:

    st.markdown(
        f"""
        <div class="ml-header">

            <div class="ml-header-content">

                <div class="ml-header-eyebrow">
                    GROUNDED RAG ASSISTANT
                </div>

                <h1>
                    🌿 {safe_text(settings.app_name)}
                </h1>

                <p>
                    {safe_text(settings.app_tagline)}
                    · answers are generated from your uploaded
                    leaflet and backed by retrieved evidence.
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# STATUS
# ============================================================================


def render_system_status() -> None:

    reranker_status = (
        "Reranker ON"
        if settings.reranker_enabled
        else "Reranker OFF"
    )

    grounding_status = (
        "Grounding check ON"
        if settings.grounding_check_enabled
        else "Grounding check OFF"
    )

    st.markdown(
        f"""
        <div class="ml-status-bar">

            <span class="ml-status ml-status-active">
                ● System ready
            </span>

            <span class="ml-status">
                🔎 {safe_text(settings.retrieval_mode)}
            </span>

            <span class="ml-status">
                ↗ {safe_text(reranker_status)}
            </span>

            <span class="ml-status">
                ✓ {safe_text(grounding_status)}
            </span>

            <span class="ml-status">
                📌 Citations enabled
            </span>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# FULL SUMMARY
# ============================================================================


def render_full_summary(
    summary: Any,
) -> None:

    st.markdown(
        "### 📋 Leaflet overview"
    )

    fields = [
        (
            "Medicine Name",
            "medicine_name",
        ),
        (
            "Drug Class",
            "drug_class",
        ),
        (
            "Uses",
            "uses",
        ),
        (
            "Who Should Not Use It",
            "who_should_not_use_it",
        ),
        (
            "Pregnancy",
            "pregnancy",
        ),
        (
            "Breastfeeding",
            "breastfeeding",
        ),
        (
            "Children",
            "children",
        ),
        (
            "Elderly",
            "elderly",
        ),
        (
            "Dosage Instructions",
            "dosage_instructions",
        ),
        (
            "Missed Dose",
            "missed_dose",
        ),
        (
            "Overdose",
            "overdose",
        ),
        (
            "Storage",
            "storage",
        ),
        (
            "Common Side Effects",
            "common_side_effects",
        ),
        (
            "Serious Side Effects",
            "serious_side_effects",
        ),
        (
            "Warnings",
            "warnings",
        ),
        (
            "When To Contact A Doctor",
            "when_to_contact_a_doctor",
        ),
    ]

    columns = st.columns(2)

    rendered = 0

    for label, attribute in fields:

        value = get_value(
            summary,
            attribute,
            "",
        )

        if not value:
            continue

        with columns[
            rendered % 2
        ]:

            st.markdown(
                f"""
                <div class="ml-summary-card">

                    <div class="ml-summary-label">
                        {safe_text(label)}
                    </div>

                    <div class="ml-summary-value">
                        {safe_text(value)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

        rendered += 1


# ============================================================================
# QUICK QUESTIONS
# ============================================================================


def render_quick_questions() -> None:

    st.markdown(
        "### 💡 Quick questions"
    )

    st.caption(
        "Choose a common question or ask anything "
        "about the uploaded leaflet."
    )

    columns = st.columns(4)

    for index, question in enumerate(
        QUICK_QUESTIONS
    ):

        if columns[
            index % 4
        ].button(
            question,
            key=f"quick_question_{index}",
            use_container_width=True,
        ):

            st.session_state.pending_question = (
                question
            )


# ============================================================================
# SOURCES
# ============================================================================


def render_sources(
    answer: StructuredAnswer,
) -> None:
    """Render exact citation provenance."""

    if not answer.sources:
        return

    with st.expander(
        f"📚 Evidence & sources "
        f"({len(answer.sources)})"
    ):

        for source in answer.sources:

            source_id = source_value(
                source,
                "chunk_id",
                "source",
            )

            page = source_value(
                source,
                "page",
                "?",
            )

            section = source_value(
                source,
                "section",
                "General",
            )

            method = source_value(
                source,
                "retrieval_method",
                "retrieval",
            )

            confidence = source_value(
                source,
                "confidence",
                "",
            )

            file_name = source_value(
                source,
                "source_file",
                "",
            )

            st.markdown(
                f"""
                <div class="ml-source">

                    <div class="ml-source-title">
                        📄 Page {safe_text(page)}
                        · {safe_text(section)}
                    </div>

                    <div class="ml-source-meta">
                        {safe_text(file_name)}
                        <br>
                        Source ID:
                        {safe_text(source_id)}
                    </div>

                    <span class="ml-badge ml-badge-muted">
                        {safe_text(method)}
                    </span>

                    {
                        f'''
                        <span class="ml-badge ml-badge-green">
                            confidence {safe_text(confidence)}
                        </span>
                        '''
                        if confidence
                        else ""
                    }

                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================================
# ANSWER QUALITY
# ============================================================================


def render_answer_quality(
    answer: StructuredAnswer,
) -> None:

    st.markdown(
        "#### 🔎 Answer quality"
    )

    retrieval_confidence = getattr(
        answer,
        "retrieval_confidence",
        0.0,
    )

    grounding_score = getattr(
        answer,
        "grounding_score",
        0.0,
    )

    grounded = getattr(
        answer,
        "grounded",
        False,
    )

    columns = st.columns(4)

    with columns[0]:

        st.metric(
            "Answer",
            f"{answer.confidence:.0f}%",
        )

    with columns[1]:

        st.metric(
            "Retrieval",
            f"{retrieval_confidence:.0f}%",
        )

    with columns[2]:

        st.metric(
            "Grounding",
            f"{grounding_score * 100:.0f}%",
        )

    with columns[3]:

        st.metric(
            "Status",
            "Grounded"
            if grounded
            else "Review",
        )

    st.progress(
        min(
            max(
                answer.confidence / 100,
                0.0,
            ),
            1.0,
        )
    )


# ============================================================================
# ANSWER CARD
# ============================================================================


def render_answer_card(
    entry: dict,
    index: int,
) -> None:

    question = entry[
        "question"
    ]

    answer: StructuredAnswer = entry[
        "answer"
    ]

    emergency_terms = entry[
        "emergency"
    ]

    # ------------------------------------------------------------------------
    # User
    # ------------------------------------------------------------------------

    with st.chat_message(
        "user",
        avatar="👤",
    ):

        st.write(
            question
        )

    # ------------------------------------------------------------------------
    # Assistant
    # ------------------------------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🌿",
    ):

        if emergency_terms:

            st.markdown(
                f"""
                <div class="ml-emergency">

                    <strong>
                        🚨 Potential emergency
                    </strong>

                    <br><br>

                    {safe_text(
                        EMERGENCY_MESSAGE
                    )}

                </div>
                """,
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # Answer
        # --------------------------------------------------------------------

        if answer.insufficient_information:

            st.warning(
                answer.answer
                or settings.insufficient_info_message
            )

        else:

            st.markdown(
                """
                <div class="ml-card">

                    <div class="ml-card-title">
                        Answer
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                answer.answer
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # Important information
        # --------------------------------------------------------------------

        if answer.important_information:

            st.markdown(
                """
                <div class="ml-card">

                    <div class="ml-card-title">
                        ℹ️ Important information
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                answer.important_information
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # Warnings
        # --------------------------------------------------------------------

        if answer.warnings:

            st.markdown(
                """
                <div
                    class="ml-card"
                    style="
                        border-left:
                        3px solid #ff7070;
                    "
                >

                    <div class="ml-card-title">
                        ⚠️ Warnings
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                answer.warnings
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # Practical information
        # --------------------------------------------------------------------

        if answer.practical_advice:

            st.markdown(
                """
                <div class="ml-card">

                    <div class="ml-card-title">
                        💡 Practical information
                    </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                answer.practical_advice
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        # --------------------------------------------------------------------
        # Quality
        # --------------------------------------------------------------------

        render_answer_quality(
            answer
        )

        # --------------------------------------------------------------------
        # Sources
        # --------------------------------------------------------------------

        render_sources(
            answer
        )

        # --------------------------------------------------------------------
        # Explanation
        # --------------------------------------------------------------------

        if answer.explanation:

            with st.expander(
                "🔍 Why did MediLeaf answer this?"
            ):

                st.write(
                    answer.explanation
                )

        # --------------------------------------------------------------------
        # Response details
        # --------------------------------------------------------------------

        with st.expander(
            "⚙️ Response details"
        ):

            detail_columns = st.columns(2)

            with detail_columns[0]:

                st.caption(
                    "Generation time"
                )

                st.write(
                    f"{answer.execution_time_seconds:.2f}s"
                )

            with detail_columns[1]:

                st.caption(
                    "Grounding status"
                )

                st.write(
                    "Verified against retrieved context"
                    if answer.grounded
                    else "Grounding check failed"
                )

        # --------------------------------------------------------------------
        # Disclaimer
        # --------------------------------------------------------------------

        st.caption(
            answer.disclaimer
        )

        # --------------------------------------------------------------------
        # Feedback
        # --------------------------------------------------------------------

        feedback_columns = st.columns(
            [1, 1, 5]
        )

        current_feedback = (
            st.session_state.feedback.get(
                index
            )
        )

        if feedback_columns[0].button(
            "👍 Helpful",
            key=f"feedback_up_{index}",
            disabled=(
                current_feedback
                is not None
            ),
        ):

            st.session_state.feedback[
                index
            ] = "up"

            log_feedback(
                st.session_state.session_id,
                question,
                True,
            )
            record_feedback(
                st.session_state.session_id,
                question,
                True,
            )

            st.rerun()

        if feedback_columns[1].button(
            "👎 Not helpful",
            key=f"feedback_down_{index}",
            disabled=(
                current_feedback
                is not None
            ),
        ):

            st.session_state.feedback[
                index
            ] = "down"

            log_feedback(
                st.session_state.session_id,
                question,
                False,
            )
            record_feedback(
                st.session_state.session_id,
                question,
                False,
            )

            st.rerun()

        if current_feedback:

            st.caption(
                "Thanks for your feedback!"
                if current_feedback == "up"
                else
                "Thanks — this feedback will help improve the system."
            )


# ============================================================================
# EMPTY STATE
# ============================================================================


def render_empty_state() -> None:

    st.markdown(
        """
        <div class="ml-empty">

            <div class="ml-empty-icon">
                🌿
            </div>

            <h2>
                Understand your medicine leaflet
            </h2>

            <p>
                Upload one or more medicine leaflet PDFs
                from the sidebar. MediLeaf parses, indexes
                and retrieves relevant sections before
                generating a grounded answer.
            </p>

            <div class="ml-empty-flow">
                PDF → Chunking → Vector + BM25
                → Reranking → Grounded Answer → Sources
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# FOOTER
# ============================================================================


def render_footer() -> None:

    st.markdown(
        """
        <div class="ml-footer">
            MediLeaf AI · Retrieval-Augmented Medicine Leaflet Assistant
            · Answers are grounded in the uploaded document.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:

    problems = settings.validate()

    render_sidebar()
    render_header()

    # ------------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------------

    if problems:

        with st.expander(
            "⚠️ Configuration issues",
            expanded=True,
        ):

            for problem in problems:
                st.error(
                    problem
                )

    # ------------------------------------------------------------------------
    # Empty state
    # ------------------------------------------------------------------------

    if st.session_state.vector_store is None:

        render_empty_state()
        render_footer()

        return

    # ------------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------------

    render_system_status()

    # ------------------------------------------------------------------------
    # Index statistics
    # ------------------------------------------------------------------------

    stats = (
        st.session_state.index_stats
    )

    if stats:

        columns = st.columns(3)

        with columns[0]:

            st.metric(
                "Documents",
                stats.get(
                    "documents",
                    0,
                ),
            )

        with columns[1]:

            st.metric(
                "Indexed chunks",
                stats.get(
                    "chunks",
                    0,
                ),
            )

        with columns[2]:

            st.metric(
                "Retrieval",
                settings.retrieval_mode,
            )

    # ------------------------------------------------------------------------
    # Loaded documents
    # ------------------------------------------------------------------------

    if st.session_state.leaflet_names:

        st.caption(
            "🟢 "
            f"{len(st.session_state.leaflet_names)} "
            "document(s) ready for questions."
        )

    # ------------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------------

    if st.session_state.summary:

        with st.expander(
            "📋 View automatic leaflet summary",
            expanded=(
                len(
                    st.session_state.chat_log
                )
                == 0
            ),
        ):

            render_full_summary(
                st.session_state.summary
            )

    # ------------------------------------------------------------------------
    # Quick questions
    # ------------------------------------------------------------------------

    render_quick_questions()

    st.divider()

    # ------------------------------------------------------------------------
    # Chat history
    # ------------------------------------------------------------------------

    for index, entry in enumerate(
        st.session_state.chat_log
    ):

        render_answer_card(
            entry,
            index,
        )

    # ------------------------------------------------------------------------
    # Chat input
    # ------------------------------------------------------------------------

    typed_question = st.chat_input(
        "Ask anything about this medicine leaflet..."
    )

    question_to_run = None

    if st.session_state.pending_question:

        question_to_run = (
            st.session_state.pending_question
        )

        st.session_state.pending_question = None

    elif typed_question:

        question_to_run = typed_question

    if question_to_run:

        handle_question(
            question_to_run
        )

        st.rerun()

    render_footer()


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    main()