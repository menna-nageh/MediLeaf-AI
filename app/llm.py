"""
llm.py
------
Gemini LLM wrapper for MediLeaf AI.

Responsibilities:
- Generate grounded answers from retrieved leaflet chunks.
- Preserve exact source provenance for citations.
- Separate retrieval confidence from answer confidence.
- Validate generated citations against retrieved chunks.
- Run a lightweight grounding safety check.
- Generate structured leaflet summaries.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.prompt_builder import (
    build_answer_prompt,
    build_summary_prompt,
    normalize_answer_response,
    normalize_summary_response,
    safe_parse_json,
)
from app.retriever import RetrievedChunk


logger = logging.getLogger(__name__)


# ============================================================================
# Errors
# ============================================================================


class LLMError(Exception):
    """Raised when the LLM cannot generate a valid response."""


# ============================================================================
# Structured outputs
# ============================================================================


@dataclass
class SourceCitation:
    """Exact provenance information for a retrieved source chunk."""

    source_file: str
    page: int
    section: str
    chunk_id: str = ""
    retrieval_method: str = ""
    confidence: float = 0.0


@dataclass
class StructuredAnswer:
    """Final structured answer returned by the generation layer."""

    insufficient_information: bool
    answer: str
    important_information: str
    warnings: str
    practical_advice: str
    explanation: str

    sources: list[SourceCitation] = field(
        default_factory=list
    )

    retrieval_confidence: float = 0.0

    grounded: bool = True
    grounding_score: float = 1.0

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


# ============================================================================
# LLM instance
# ============================================================================


_llm_instance: ChatGoogleGenerativeAI | None = None


def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the cached Gemini LLM instance.
    """

    global _llm_instance

    if _llm_instance is None:

        if not settings.google_api_key:
            raise LLMError(
                "Missing GOOGLE_API_KEY. "
                "Add it to the .env file."
            )

        try:
            _llm_instance = ChatGoogleGenerativeAI(
                model=settings.llm_model_name,
                temperature=settings.llm_temperature,
                max_output_tokens=(
                    settings.llm_max_output_tokens
                ),
                google_api_key=(
                    settings.google_api_key
                ),
            )

        except Exception as exc:
            raise LLMError(
                f"Failed to initialize Gemini: {exc}"
            ) from exc

    return _llm_instance


# ============================================================================
# LLM invocation
# ============================================================================


def _invoke(prompt: str) -> str:
    """
    Send a prompt to Gemini and normalize its response.
    """

    if not prompt or not prompt.strip():
        raise LLMError(
            "Cannot invoke LLM with an empty prompt."
        )

    try:
        llm = get_llm()

        response = llm.invoke(
            [
                HumanMessage(
                    content=prompt
                )
            ]
        )

        content = response.content

        if isinstance(
            content,
            list,
        ):
            parts: list[str] = []

            for block in content:

                if isinstance(
                    block,
                    dict,
                ):
                    text = block.get(
                        "text",
                        "",
                    )

                    if text:
                        parts.append(
                            str(text)
                        )

                else:
                    parts.append(
                        str(block)
                    )

            content = "".join(
                parts
            )

        text = str(
            content
        ).strip()

        if not text:
            raise LLMError(
                "The LLM returned an empty response."
            )

        return text

    except LLMError:
        raise

    except Exception as exc:
        logger.exception(
            "Gemini invocation failed."
        )

        raise LLMError(
            f"Gemini request failed: {exc}"
        ) from exc


# ============================================================================
# Citation handling
# ============================================================================


def _build_source_citations(
    chunks: list[RetrievedChunk],
) -> list[SourceCitation]:
    """
    Convert retrieved chunks into structured citation objects.
    """

    citations: list[SourceCitation] = []

    seen: set[str] = set()

    for chunk in chunks:

        source_id = (
            chunk.chunk_id
            or (
                f"{chunk.source_file}:"
                f"{chunk.page_number}:"
                f"{chunk.section}"
            )
        )

        if source_id in seen:
            continue

        seen.add(
            source_id
        )

        citations.append(
            SourceCitation(
                source_file=chunk.source_file,
                page=chunk.page_number,
                section=chunk.section,
                chunk_id=chunk.chunk_id,
                retrieval_method=(
                    chunk.retrieval_method
                ),
                confidence=round(
                    chunk.confidence,
                    1,
                ),
            )
        )

    return citations


def _extract_citation_ids(
    parsed_answer: dict,
) -> list[str]:
    """
    Extract citation IDs returned by the model.
    """

    raw = parsed_answer.get(
        "citations",
        [],
    )

    if not isinstance(
        raw,
        list,
    ):
        return []

    citation_ids: list[str] = []

    for item in raw:

        if not isinstance(
            item,
            str,
        ):
            continue

        value = item.strip()

        if value:
            citation_ids.append(
                value
            )

    return citation_ids


def _validate_citations(
    citation_ids: list[str],
    chunks: list[RetrievedChunk],
) -> tuple[bool, float]:
    """
    Validate that every citation refers to retrieved context.

    Returns:
        (is_valid, citation_coverage)
    """

    valid_ids = {
        chunk.chunk_id
        for chunk in chunks
        if chunk.chunk_id
    }

    if not citation_ids:
        return False, 0.0

    if not valid_ids:
        return False, 0.0

    unique_requested = set(
        citation_ids
    )

    valid_citations = (
        unique_requested
        & valid_ids
    )

    coverage = (
        len(valid_citations)
        / max(
            len(unique_requested),
            1,
        )
    )

    is_valid = (
        valid_citations
        == unique_requested
    )

    return (
        is_valid,
        round(
            coverage,
            3,
        ),
    )


def _filter_cited_sources(
    sources: list[SourceCitation],
    citation_ids: list[str],
) -> list[SourceCitation]:
    """
    Return only sources explicitly cited by the model.
    """

    if not citation_ids:
        return []

    cited_ids = set(
        citation_ids
    )

    return [
        source
        for source in sources
        if source.chunk_id
        and source.chunk_id in cited_ids
    ]


# ============================================================================
# Grounding verification
# ============================================================================


def _normalize_for_grounding(
    text: str,
) -> str:
    """
    Normalize text for lightweight lexical overlap.
    """

    if not text:
        return ""

    text = text.lower()

    text = re.sub(
        r"[^\w\u0600-\u06FF\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def _grounding_check(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[bool, float]:
    """
    Lightweight lexical grounding check.

    This is a safety heuristic, NOT an LLM-as-a-judge evaluator.
    """

    if not answer.strip():
        return False, 0.0

    if not chunks:
        return False, 0.0

    answer_normalized = (
        _normalize_for_grounding(
            answer
        )
    )

    answer_tokens = set(
        answer_normalized.split()
    )

    if len(answer_tokens) < 3:
        return True, 1.0

    context = " ".join(
        chunk.text
        for chunk in chunks
        if chunk.text
    )

    context_normalized = (
        _normalize_for_grounding(
            context
        )
    )

    context_tokens = set(
        context_normalized.split()
    )

    if not context_tokens:
        return False, 0.0

    overlap = (
        len(
            answer_tokens
            & context_tokens
        )
        / len(answer_tokens)
    )

    grounded = (
        overlap
        >= settings.grounding_min_overlap
        if hasattr(
            settings,
            "grounding_min_overlap",
        )
        else overlap >= 0.25
    )

    return (
        grounded,
        round(
            min(
                1.0,
                overlap,
            ),
            3,
        ),
    )


# ============================================================================
# Answer generation
# ============================================================================


def answer_question(
    question: str,
    chunks: list[RetrievedChunk],
    memory_context: str = "",
    overall_confidence: float = 0.0,
    prompt_version: str = "v2",
) -> StructuredAnswer:
    """
    Generate a grounded structured answer.

    Pipeline:

        Question
            ↓
        Retrieval
            ↓
        Reranking
            ↓
        Prompt Construction
            ↓
        Gemini
            ↓
        JSON Parsing
            ↓
        Citation Validation
            ↓
        Grounding Check
            ↓
        StructuredAnswer
    """

    start = time.perf_counter()

    # -----------------------------------------------------------------------
    # Empty question
    # -----------------------------------------------------------------------

    if not question or not question.strip():

        return StructuredAnswer(
            insufficient_information=True,
            answer=(
                settings.insufficient_info_message
            ),
            important_information="",
            warnings="",
            practical_advice="",
            explanation="Empty question.",
            sources=[],
            retrieval_confidence=0.0,
            grounded=False,
            grounding_score=0.0,
            confidence=0.0,
            execution_time_seconds=round(
                time.perf_counter()
                - start,
                3,
            ),
        )

    # -----------------------------------------------------------------------
    # No retrieved context
    # -----------------------------------------------------------------------

    if not chunks:

        return StructuredAnswer(
            insufficient_information=True,
            answer=(
                settings.insufficient_info_message
            ),
            important_information="",
            warnings="",
            practical_advice="",
            explanation=(
                "No sufficiently relevant leaflet "
                "context was retrieved."
            ),
            sources=[],
            retrieval_confidence=(
                overall_confidence
            ),
            grounded=True,
            grounding_score=1.0,
            confidence=round(
                max(
                    0.0,
                    min(
                        100.0,
                        overall_confidence,
                    ),
                )
                * 0.4,
                1,
            ),
            execution_time_seconds=round(
                time.perf_counter()
                - start,
                3,
            ),
        )

    # -----------------------------------------------------------------------
    # Build canonical prompt
    # -----------------------------------------------------------------------

    prompt = build_answer_prompt(
        question=question,
        chunks=chunks,
        memory_context=memory_context,
        prompt_version=prompt_version,
    )

    raw = _invoke(
        prompt
    )

    # -----------------------------------------------------------------------
    # Parse JSON
    # -----------------------------------------------------------------------

    try:

        parsed = safe_parse_json(
            raw
        )

        parsed = normalize_answer_response(
            parsed
        )

    except Exception as exc:

        logger.warning(
            "Could not parse Gemini answer: %s",
            exc,
        )

        return StructuredAnswer(
            insufficient_information=True,
            answer=(
                "I could not safely process the "
                "AI model's response."
            ),
            important_information="",
            warnings="",
            practical_advice="",
            explanation=(
                "The model did not return valid "
                "structured JSON."
            ),
            sources=[],
            retrieval_confidence=(
                overall_confidence
            ),
            grounded=False,
            grounding_score=0.0,
            confidence=0.0,
            execution_time_seconds=round(
                time.perf_counter()
                - start,
                3,
            ),
        )

    # -----------------------------------------------------------------------
    # Extract fields
    # -----------------------------------------------------------------------

    insufficient = bool(
        parsed.get(
            "insufficient_information",
            False,
        )
    )

    answer = str(
        parsed.get(
            "answer",
            "",
        )
    ).strip()

    important_information = str(
        parsed.get(
            "important_information",
            "",
        )
    ).strip()

    warnings = str(
        parsed.get(
            "warnings",
            "",
        )
    ).strip()

    practical_advice = str(
        parsed.get(
            "practical_advice",
            "",
        )
    ).strip()

    explanation = str(
        parsed.get(
            "explanation",
            "",
        )
    ).strip()

    # -----------------------------------------------------------------------
    # Citation validation
    # -----------------------------------------------------------------------

    citation_ids = _extract_citation_ids(
        parsed
    )

    citations_valid, citation_coverage = (
        _validate_citations(
            citation_ids,
            chunks,
        )
    )

    all_sources = _build_source_citations(
        chunks
    )

    selected_sources = (
        _filter_cited_sources(
            all_sources,
            citation_ids,
        )
    )

    # A grounded answer with citation support
    # should expose only the sources explicitly cited.
    if (
        settings.citation_enabled
        and not insufficient
        and not citations_valid
    ):
        selected_sources = []

    # -----------------------------------------------------------------------
    # Grounding verification
    # -----------------------------------------------------------------------

    if insufficient:

        grounded = True
        grounding_score = 1.0

    elif settings.grounding_check_enabled:

        grounded, grounding_score = (
            _grounding_check(
                answer,
                chunks,
            )
        )

    else:

        grounded = True
        grounding_score = 1.0

    # -----------------------------------------------------------------------
    # Citation quality
    # -----------------------------------------------------------------------

    if (
        not insufficient
        and settings.citation_enabled
    ):

        if not citation_ids:
            grounded = False
            grounding_score *= 0.5

        elif not citations_valid:
            grounded = False
            grounding_score *= 0.5

        else:
            grounding_score *= (
                0.5
                + 0.5
                * citation_coverage
            )

    grounding_score = round(
        max(
            0.0,
            min(
                1.0,
                grounding_score,
            ),
        ),
        3,
    )

    # -----------------------------------------------------------------------
    # Confidence calculation
    # -----------------------------------------------------------------------

    retrieval_confidence = max(
        0.0,
        min(
            100.0,
            float(
                overall_confidence
            ),
        ),
    )

    retrieval_component = (
        retrieval_confidence
        / 100.0
    )

    confidence = round(
        (
            retrieval_component * 0.60
            + grounding_score * 0.40
        )
        * 100.0,
        1,
    )

    # Never expose high confidence when grounding failed.
    if not grounded:
        confidence = min(
            confidence,
            35.0,
        )

    # -----------------------------------------------------------------------
    # Safety fallback
    # -----------------------------------------------------------------------

    if not insufficient and not grounded:

        return StructuredAnswer(
            insufficient_information=True,
            answer=(
                settings.insufficient_info_message
            ),
            important_information="",
            warnings="",
            practical_advice="",
            explanation=(
                "The generated answer could not be "
                "sufficiently grounded in the retrieved "
                "leaflet sources."
            ),
            sources=[],
            retrieval_confidence=round(
                retrieval_confidence,
                1,
            ),
            grounded=False,
            grounding_score=grounding_score,
            confidence=confidence,
            execution_time_seconds=round(
                time.perf_counter()
                - start,
                3,
            ),
        )

    # -----------------------------------------------------------------------
    # Empty answer safety
    # -----------------------------------------------------------------------

    if (
        not insufficient
        and not answer
    ):

        return StructuredAnswer(
            insufficient_information=True,
            answer=(
                settings.insufficient_info_message
            ),
            important_information="",
            warnings="",
            practical_advice="",
            explanation=(
                "The model returned no usable answer."
            ),
            sources=[],
            retrieval_confidence=round(
                retrieval_confidence,
                1,
            ),
            grounded=False,
            grounding_score=0.0,
            confidence=0.0,
            execution_time_seconds=round(
                time.perf_counter()
                - start,
                3,
            ),
        )

    # -----------------------------------------------------------------------
    # Final structured response
    # -----------------------------------------------------------------------

    return StructuredAnswer(
        insufficient_information=(
            insufficient
        ),
        answer=answer,
        important_information=(
            important_information
        ),
        warnings=warnings,
        practical_advice=(
            practical_advice
        ),
        explanation=explanation,
        sources=selected_sources,
        retrieval_confidence=round(
            retrieval_confidence,
            1,
        ),
        grounded=grounded,
        grounding_score=grounding_score,
        confidence=confidence,
        execution_time_seconds=round(
            time.perf_counter()
            - start,
            3,
        ),
    )


# ============================================================================
# Summary generation
# ============================================================================


def generate_summary(
    leaflet_text: str,
) -> LeafletSummary:
    """
    Generate a structured summary from leaflet text.
    """

    if not leaflet_text or not leaflet_text.strip():
        return LeafletSummary()

    prompt = build_summary_prompt(
        leaflet_text
    )

    raw = _invoke(
        prompt
    )

    try:

        parsed = safe_parse_json(
            raw
        )

        parsed = normalize_summary_response(
            parsed
        )

    except Exception as exc:

        logger.warning(
            "Could not parse leaflet summary: %s",
            exc,
        )

        return LeafletSummary()

    try:

        return LeafletSummary(
            **parsed
        )

    except TypeError as exc:

        logger.warning(
            "Invalid leaflet summary structure: %s",
            exc,
        )

        return LeafletSummary()