"""
llm.py
------
Gemini wrapper for MediLeaf AI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.retriever import RetrievedChunk
from app.prompt_builder import (
    build_answer_prompt,
    build_summary_prompt,
    safe_parse_json,
)


class LLMError(Exception):
    pass


@dataclass
class StructuredAnswer:
    insufficient_information: bool
    answer: str
    important_information: str
    warnings: str
    practical_advice: str
    explanation: str
    sources: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    disclaimer: str = settings.disclaimer
    execution_time_seconds: float = 0.0


@dataclass
class LeafletSummary:
    medicine_name: str = ""
    drug_class: str = ""
    uses: str = ""
    who_should_not_use_it: str = ""
    pregnancy: str = ""
    breastfeeding: str = ""
    children: str = ""
    elderly: str = ""
    dosage_instructions: str = ""
    missed_dose: str = ""
    overdose: str = ""
    storage: str = ""
    common_side_effects: str = ""
    serious_side_effects: str = ""
    warnings: str = ""
    when_to_contact_a_doctor: str = ""


_llm_instance: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm_instance

    if _llm_instance is None:
        if not settings.google_api_key:
            raise LLMError("Missing GOOGLE_API_KEY")

        _llm_instance = ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            google_api_key=settings.google_api_key,
        )

    return _llm_instance


def _invoke(prompt: str) -> str:
    try:
        llm = get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])

        content = response.content

        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        text = content.strip()

        print("\n========== RAW LLM RESPONSE ==========")
        print(text)
        print("======================================\n")

        return text

    except Exception as exc:
        raise LLMError(f"Gemini request failed: {exc}") from exc


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    memory_context: str,
    overall_confidence: float,
) -> StructuredAnswer:

    start = time.perf_counter()

    # Always build prompt even with empty chunks - let the LLM use memory/decide
    prompt = build_answer_prompt(question, chunks, memory_context)

    print("\n========== PROMPT SENT ==========")
    print(prompt[:1000])
    print("================================\n")

    raw = _invoke(prompt)

    try:
        parsed = safe_parse_json(raw)
    except Exception:
        return StructuredAnswer(
            insufficient_information=True,
            answer="I could not process the response from the AI model.",
            important_information="",
            warnings="",
            practical_advice="",
            explanation="JSON parsing failed.",
            sources=[],
            confidence=0.0,
            execution_time_seconds=round(time.perf_counter() - start, 3),
        )

    insufficient = bool(parsed.get("insufficient_information", False))

    sources = []
    if not insufficient and chunks:
        sources = [
            {"page": c.page_number, "section": c.section}
            for c in chunks
        ]

    return StructuredAnswer(
        insufficient_information=insufficient,
        answer=parsed.get("answer", ""),
        important_information=parsed.get("important_information", ""),
        warnings=parsed.get("warnings", ""),
        practical_advice=parsed.get("practical_advice", ""),
        explanation=parsed.get("explanation", ""),
        sources=sources,
        confidence=overall_confidence,
        execution_time_seconds=round(time.perf_counter() - start, 3),
    )


def generate_summary(leaflet_text: str) -> LeafletSummary:
    prompt = build_summary_prompt(leaflet_text)
    raw = _invoke(prompt)

    try:
        parsed = safe_parse_json(raw)
    except Exception:
        return LeafletSummary()

    known_fields = LeafletSummary.__dataclass_fields__.keys()
    filtered = {k: v for k, v in parsed.items() if k in known_fields}

    return LeafletSummary(**filtered)

