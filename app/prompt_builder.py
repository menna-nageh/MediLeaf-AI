"""
prompt_builder.py
-----------------
Centralized prompt engineering for MediLeaf AI.

Responsibilities:
- Strict context-grounded RAG answering
- Citation-aware prompts
- Conversation-memory separation
- Structured JSON output
- Safe JSON parsing
- Context-only leaflet summarization
"""

from __future__ import annotations

import json
from typing import Any

from app.retriever import RetrievedChunk


# ============================================================================
# SYSTEM RULES
# ============================================================================


SYSTEM_RULES = """
You are MediLeaf AI, a retrieval-augmented assistant for understanding
medical leaflets.

Your answers MUST be grounded ONLY in the CURRENT CONTEXT.

ABSOLUTE RULES:

1. Use ONLY information explicitly supported by CURRENT CONTEXT.
2. Do NOT use pretrained knowledge, general medical knowledge, or outside
   information to fill missing details.
3. NEVER guess, infer unsupported medical facts, or fabricate information.
4. If the context does not clearly support the answer, set
   "insufficient_information" to true.
5. Carefully distinguish the medicine asked about from other medicines,
   conditions, or examples mentioned in the leaflet.
6. Every factual medical claim must be supported by one or more SOURCE_IDs
   from CURRENT CONTEXT.
7. Citation IDs MUST be copied exactly from CURRENT CONTEXT.
8. NEVER invent, modify, shorten, or create citation IDs.
9. Do not cite irrelevant sources.
10. Previous conversation is NOT medical evidence.
11. Do not use previous conversation to introduce or confirm medical facts.
12. Respond in the SAME LANGUAGE as the user's question.
13. Return ONLY valid JSON.
14. Do not return Markdown, code fences, or text outside the JSON object.
15. If uncertain whether the context supports a claim, prefer
    "insufficient_information": true.
16. Do not provide medical advice beyond what is explicitly stated in
    the leaflet.
"""


# ============================================================================
# JSON SCHEMAS
# ============================================================================


ANSWER_JSON_SCHEMA = """{
  "insufficient_information": false,
  "answer": "",
  "important_information": "",
  "warnings": "",
  "practical_advice": "",
  "explanation": "",
  "citations": []
}"""


SUMMARY_JSON_SCHEMA = """{
  "medicine_name": "",
  "drug_class": "",
  "uses": "",
  "who_should_not_use_it": "",
  "pregnancy": "",
  "breastfeeding": "",
  "children": "",
  "elderly": "",
  "dosage_instructions": "",
  "missed_dose": "",
  "overdose": "",
  "storage": "",
  "common_side_effects": "",
  "serious_side_effects": "",
  "warnings": "",
  "when_to_contact_a_doctor": ""
}"""


ANSWER_FIELDS = (
    "insufficient_information",
    "answer",
    "important_information",
    "warnings",
    "practical_advice",
    "explanation",
    "citations",
)


SUMMARY_FIELDS = (
    "medicine_name",
    "drug_class",
    "uses",
    "who_should_not_use_it",
    "pregnancy",
    "breastfeeding",
    "children",
    "elderly",
    "dosage_instructions",
    "missed_dose",
    "overdose",
    "storage",
    "common_side_effects",
    "serious_side_effects",
    "warnings",
    "when_to_contact_a_doctor",
)


# ============================================================================
# CONTEXT FORMATTER
# ============================================================================


def _format_context(
    chunks: list[RetrievedChunk],
) -> str:
    """
    Convert retrieved chunks into a citation-aware context block.

    Each chunk receives a stable SOURCE_ID that the LLM can reference
    in the citations field.
    """

    if not chunks:
        return "[NO RETRIEVED CONTEXT]"

    parts: list[str] = []

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        source_id = (
            chunk.chunk_id
            or f"source_{index}"
        )

        source_file = str(
            getattr(
                chunk,
                "source_file",
                "",
            )
        )

        page_number = getattr(
            chunk,
            "page_number",
            0,
        )

        section = str(
            getattr(
                chunk,
                "section",
                "General",
            )
        )

        retrieval_method = str(
            getattr(
                chunk,
                "retrieval_method",
                "",
            )
        )

        confidence = float(
            getattr(
                chunk,
                "confidence",
                0.0,
            )
        )

        content = str(
            getattr(
                chunk,
                "text",
                "",
            )
        ).strip()

        if not content:
            continue

        parts.append(
            "\n".join(
                [
                    f"SOURCE_ID: {source_id}",
                    f"FILE: {source_file}",
                    f"PAGE: {page_number}",
                    f"SECTION: {section}",
                    f"RETRIEVAL_METHOD: {retrieval_method}",
                    f"RETRIEVAL_CONFIDENCE: {confidence:.2f}",
                    "CONTENT:",
                    content,
                    "END_SOURCE",
                ]
            )
        )

    if not parts:
        return "[NO RETRIEVED CONTEXT]"

    return "\n\n---\n\n".join(parts)


# ============================================================================
# ANSWER PROMPT
# ============================================================================


def build_answer_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    memory_context: str = "",
    prompt_version: str = "v2",
) -> str:
    """
    Build the canonical grounded RAG prompt.

    Conversation memory is provided ONLY for resolving references in
    follow-up questions. It is never considered medical evidence.
    """

    question = question.strip()

    context_block = _format_context(
        chunks
    )

    memory_block = ""

    if memory_context and memory_context.strip():
        memory_block = f"""
============================================================
PREVIOUS CONVERSATION
============================================================

{memory_context.strip()}

MEMORY USAGE RULE:

Previous conversation may help identify what the user is referring to,
especially for follow-up questions.

However, previous conversation is NOT evidence.

You MUST NOT use memory to introduce, confirm, infer, or supplement
medical facts.

All medical facts MUST come exclusively from CURRENT CONTEXT.
"""

    version_rules = ""
    if prompt_version == "v1":
        version_rules = "Use concise JSON fields and cite every factual claim."
    elif prompt_version == "v2":
        version_rules = (
            "Before writing JSON, verify each claim against a SOURCE_ID; "
            "prefer insufficient_information over an unsupported claim."
        )
    else:
        raise ValueError("prompt_version must be 'v1' or 'v2'.")

    return f"""
{SYSTEM_RULES}

PROMPT_VERSION: {prompt_version}
{version_rules}

============================================================
CURRENT CONTEXT
============================================================

The following retrieved leaflet chunks are the ONLY medical evidence
available to you.

{context_block}

{memory_block}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER REQUIREMENTS
============================================================

Return ONLY one valid JSON object.

Use exactly this structure:

{ANSWER_JSON_SCHEMA}

============================================================
CITATION RULES
============================================================

The "citations" field:

- MUST be a JSON array.
- MUST contain SOURCE_ID strings only.
- MUST use SOURCE_ID values exactly as they appear in CURRENT CONTEXT.
- MUST cite only sources that directly support the answer.
- MUST NOT contain page numbers instead of SOURCE_IDs.
- MUST NOT contain URLs.
- MUST NOT contain invented identifiers.
- MUST NOT cite PREVIOUS CONVERSATION.
- MUST NOT cite irrelevant chunks.

If the answer contains multiple factual claims, cite every source needed
to support those claims.

============================================================
INSUFFICIENT INFORMATION
============================================================

If CURRENT CONTEXT does not clearly contain enough information:

- Set "insufficient_information" to true.
- Clearly state that the information is not available in the leaflet.
- Do NOT guess.
- Do NOT use outside medical knowledge.
- Do NOT invent dosage, side effects, warnings, interactions,
  contraindications, or recommendations.
- Use citations ONLY if a retrieved source genuinely supports the statement.
- Keep unsupported fields empty.

============================================================
FIELD RULES
============================================================

"answer":
Give the direct answer to the user's question.

"important_information":
Include important leaflet information directly relevant to the question.

"warnings":
Include ONLY warnings explicitly supported by CURRENT CONTEXT.

"practical_advice":
Include ONLY practical instructions explicitly stated in the leaflet.

"explanation":
Give a concise explanation based only on CURRENT CONTEXT.

"citations":
List the SOURCE_ID values supporting the factual content.

============================================================
LANGUAGE RULE
============================================================

Write all natural-language fields in the SAME LANGUAGE as the user's
question.

============================================================
FINAL CHECK
============================================================

Before returning the response:

1. Verify every factual claim against CURRENT CONTEXT.
2. Verify every citation exists exactly in CURRENT CONTEXT.
3. Remove unsupported claims.
4. If evidence is insufficient, set "insufficient_information" to true.
5. Return JSON ONLY.

No Markdown.
No code fences.
No commentary.
""".strip()


# ============================================================================
# SUMMARY PROMPT
# ============================================================================


def build_summary_prompt(
    leaflet_text: str,
) -> str:
    """
    Build a context-only structured leaflet summarization prompt.

    The output schema exactly matches LeafletSummary in llm.py.
    """

    leaflet_text = (
        leaflet_text.strip()
        if leaflet_text
        else ""
    )

    return f"""
You are MediLeaf AI.

Your task is to create a structured summary of the medical leaflet
provided below.

============================================================
SOURCE OF TRUTH
============================================================

The LEAFLET TEXT below is the ONLY source of information.

You MUST NOT use outside medical knowledge.

============================================================
STRICT RULES
============================================================

1. Use ONLY information explicitly present in the leaflet.
2. Do NOT guess or infer missing information.
3. Do NOT invent medicine properties.
4. Do NOT fill missing fields using general medical knowledge.
5. Preserve important safety warnings.
6. Preserve dosage information exactly as supported by the leaflet.
7. If information for a field is unavailable, return an empty string.
8. Do not add recommendations that are not explicitly stated.
9. Do not convert uncertain information into certainty.
10. Return ONLY valid JSON.
11. Do not use Markdown or code fences.
12. Keep the content in the SAME LANGUAGE as the leaflet whenever possible.
13. Keep each field concise while preserving important safety information.

============================================================
LEAFLET TEXT
============================================================

{leaflet_text}

============================================================
OUTPUT FORMAT
============================================================

Return ONLY a JSON object using exactly this structure:

{SUMMARY_JSON_SCHEMA}

============================================================
FIELD DEFINITIONS
============================================================

medicine_name:
The medicine or product name if explicitly stated.

drug_class:
The drug class or therapeutic category if explicitly stated.

uses:
Uses or indications explicitly stated in the leaflet.

who_should_not_use_it:
Contraindications or people who should not use the medicine.

pregnancy:
Information about pregnancy.

breastfeeding:
Information about breastfeeding.

children:
Information concerning children or pediatric use.

elderly:
Information concerning elderly patients.

dosage_instructions:
Dosage and administration instructions.

missed_dose:
Instructions concerning a missed dose.

overdose:
Information concerning overdose.

storage:
Storage conditions.

common_side_effects:
Common or frequently reported side effects.

serious_side_effects:
Serious side effects or severe reactions.

warnings:
Important warnings and precautions.

when_to_contact_a_doctor:
Explicit situations where the leaflet says to seek medical attention.

============================================================
FINAL VALIDATION
============================================================

Every non-empty field must be supported directly by the leaflet.

If a field is not supported by the leaflet:
return "" for that field.

Return JSON ONLY.
""".strip()


# ============================================================================
# JSON NORMALIZATION
# ============================================================================


def _normalize_answer(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize an LLM answer object.

    Unknown fields are removed and missing fields receive safe defaults.
    """

    normalized = {
        "insufficient_information": bool(
            data.get(
                "insufficient_information",
                False,
            )
        ),
        "answer": str(
            data.get(
                "answer",
                "",
            )
            or ""
        ),
        "important_information": str(
            data.get(
                "important_information",
                "",
            )
            or ""
        ),
        "warnings": str(
            data.get(
                "warnings",
                "",
            )
            or ""
        ),
        "practical_advice": str(
            data.get(
                "practical_advice",
                "",
            )
            or ""
        ),
        "explanation": str(
            data.get(
                "explanation",
                "",
            )
            or ""
        ),
        "citations": [],
    }

    raw_citations = data.get(
        "citations",
        [],
    )

    if isinstance(
        raw_citations,
        list,
    ):
        normalized["citations"] = [
            str(item).strip()
            for item in raw_citations
            if isinstance(item, str)
            and item.strip()
        ]

    return normalized


def _normalize_summary(
    data: dict[str, Any],
) -> dict[str, str]:
    """
    Normalize a structured leaflet summary.

    Every expected field is returned as a string.
    """

    normalized: dict[str, str] = {}

    for field in SUMMARY_FIELDS:

        value = data.get(
            field,
            "",
        )

        if value is None:
            value = ""

        if isinstance(
            value,
            list,
        ):
            value = "; ".join(
                str(item)
                for item in value
            )

        elif isinstance(
            value,
            dict,
        ):
            value = json.dumps(
                value,
                ensure_ascii=False,
            )

        normalized[field] = str(
            value
        ).strip()

    return normalized


# ============================================================================
# SAFE JSON PARSER
# ============================================================================


def safe_parse_json(
    raw_response: str,
) -> dict:
    """
    Safely parse a JSON response from the LLM.

    Handles:
    - Normal JSON
    - JSON wrapped in Markdown code fences
    - Responses containing surrounding text
    - Known MediLeaf answer/summary schemas
    """

    if not raw_response:
        raise ValueError(
            "Empty LLM response."
        )

    text = str(
        raw_response
    ).strip()

    # ------------------------------------------------------------------------
    # Remove Markdown code fences.
    # ------------------------------------------------------------------------

    if text.startswith("```"):

        lines = text.splitlines()

        if (
            lines
            and lines[0]
            .strip()
            .startswith("```")
        ):
            lines = lines[1:]

        if (
            lines
            and lines[-1]
            .strip()
            == "```"
        ):
            lines = lines[:-1]

        text = "\n".join(
            lines
        ).strip()

        if text.lower().startswith(
            "json"
        ):
            text = text[4:].strip()

    # ------------------------------------------------------------------------
    # First attempt: parse the complete response.
    # ------------------------------------------------------------------------

    parsed: Any

    try:
        parsed = json.loads(
            text
        )

    except json.JSONDecodeError:
        parsed = None

    if isinstance(
        parsed,
        dict,
    ):
        return parsed

    # ------------------------------------------------------------------------
    # Fallback: locate the outer JSON object.
    # ------------------------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise ValueError(
            "Invalid JSON response from LLM."
        )

    candidate = text[
        start : end + 1
    ]

    try:
        parsed = json.loads(
            candidate
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Invalid JSON response from LLM."
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise ValueError(
            "LLM response must be a JSON object."
        )

    return parsed


# ============================================================================
# PUBLIC NORMALIZATION HELPERS
# ============================================================================


def normalize_answer_response(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Public helper for normalizing an answer response.
    """

    return _normalize_answer(
        data
    )


def normalize_summary_response(
    data: dict[str, Any],
) -> dict[str, str]:
    """
    Public helper for normalizing a summary response.
    """

    return _normalize_summary(
        data
    )