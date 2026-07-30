"""
prompt_builder.py
------------------
All prompt engineering for MediLeaf AI lives here.
"""

from __future__ import annotations

import json

from app.config import settings
from app.retriever import RetrievedChunk


# =========================
# SYSTEM RULES 
# =========================
SYSTEM_RULES = f"""You are MediLeaf AI, a RAG assistant that answers ONLY from provided leaflet text.

ABSOLUTE RULES (never violate):
1. Look ONLY at the CONTEXT provided below. Do not use any pre-trained knowledge.
2. If the CONTEXT is empty or does not contain the answer, set insufficient_information to true.
3. Never generate generic drug facts from memory.
4. Never guess. Never make up information.
5. If the context contains "paracetamol is used for pain" but the question is about a DIFFERENT medicine, answer insufficient_information.
6. Return ONLY valid JSON. No extra text.
7. Respond in the SAME LANGUAGE as the question.
8. If you are unsure, set insufficient_information to true.
"""


ANSWER_JSON_SCHEMA = """{
  "insufficient_information": false,
  "answer": "",
  "important_information": "",
  "warnings": "",
  "practical_advice": "",
  "explanation": ""
}"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "(no context)"

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Page {chunk.page_number} | {chunk.section}]\n{chunk.text}"
        )

    return "\n\n".join(parts)


def build_answer_prompt(question: str, chunks: list[RetrievedChunk], memory_context: str) -> str:
    context_block = _format_context(chunks)

    return f"""{SYSTEM_RULES}

CRITICAL:
- Use ONLY the context below
- If answer not found → insufficient_information = true

CONTEXT:
{context_block}

QUESTION:
{question}

Return ONLY JSON:
{ANSWER_JSON_SCHEMA}
"""


def build_summary_prompt(leaflet_text: str) -> str:
    return f"""
Summarize this leaflet:

{leaflet_text}

Return JSON.
"""


def safe_parse_json(raw_response: str) -> dict:
    text = raw_response.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])

    raise ValueError("Invalid JSON")