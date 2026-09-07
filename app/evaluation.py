"""Offline retrieval and prompt evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.llm import StructuredAnswer, answer_question
from app.retriever import RetrievedChunk, retrieve


@dataclass(frozen=True)
class RetrievalExample:
    question: str
    relevant_chunk_ids: frozenset[str]


def hit_rate_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int = 5,
) -> float:
    if not relevant_ids:
        return 0.0
    hits = sum(
        bool(set(result[:k]) & relevant)
        for result, relevant in zip(retrieved_ids, relevant_ids)
    )
    return round(hits / len(relevant_ids), 4)


def mrr_at_k(
    retrieved_ids: list[list[str]],
    relevant_ids: list[set[str]],
    k: int = 5,
) -> float:
    if not relevant_ids:
        return 0.0
    reciprocal_ranks = []
    for result, relevant in zip(retrieved_ids, relevant_ids):
        rank = next(
            (index for index, chunk_id in enumerate(result[:k], start=1)
             if chunk_id in relevant),
            0,
        )
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4)


def evaluate_retrieval(
    store: Any,
    examples: list[RetrievalExample],
    modes: tuple[str, ...] = ("vector", "bm25", "hybrid", "hybrid_rerank"),
    k: int = 5,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    relevant = [set(example.relevant_chunk_ids) for example in examples]
    for mode in modes:
        results = [
            retrieve(store, example.question, top_k=k, mode=mode)
            for example in examples
        ]
        ids = [[chunk.chunk_id for chunk in result] for result in results]
        report[mode] = {
            f"hit_rate@{k}": hit_rate_at_k(ids, relevant, k),
            f"mrr@{k}": mrr_at_k(ids, relevant, k),
        }
    return report


def evaluate_prompts(
    questions: list[tuple[str, list[RetrievedChunk]]],
    prompt_versions: tuple[str, ...] = ("v1", "v2"),
    generator: Callable[..., StructuredAnswer] = answer_question,
) -> dict[str, dict[str, float]]:
    """Compare prompt versions using grounding and citation outcomes."""
    report: dict[str, dict[str, float]] = {}
    for version in prompt_versions:
        answers = [
            generator(question, chunks, prompt_version=version)
            for question, chunks in questions
        ]
        count = len(answers) or 1
        report[version] = {
            "grounded_rate": round(
                sum(answer.grounded for answer in answers) / count,
                4,
            ),
            "citation_rate": round(
                sum(bool(answer.sources) for answer in answers) / count,
                4,
            ),
            "mean_confidence": round(
                sum(answer.confidence for answer in answers) / count,
                4,
            ),
        }
    return report