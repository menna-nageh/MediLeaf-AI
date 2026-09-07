"""
retriever.py
------------
Retrieval layer for MediLeaf AI.

Supports:
- Vector retrieval with ChromaDB
- BM25 keyword retrieval
- Hybrid retrieval using weighted Reciprocal Rank Fusion (RRF)
- Optional Cross-Encoder reranking
- Source metadata preservation
- Retrieval confidence and context validation
- Cached BM25 corpus for faster repeated queries

The public build/load functions intentionally keep returning Chroma
so existing MediLeaf ingestion code remains compatible.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from app.config import VECTOR_DB_DIR, settings
from app.embeddings import get_embedding_function
from app.parser import LeafletChunk


logger = logging.getLogger(__name__)


# ============================================================================
# Retrieved result
# ============================================================================


@dataclass
class RetrievedChunk:
    """A single retrieved leaflet chunk with ranking information."""

    text: str
    page_number: int
    section: str
    source_file: str
    confidence: float

    chunk_id: str = ""
    retrieval_method: str = "vector"

    # Raw reranker score when reranking is used.
    rerank_score: float | None = None

    # Original retrieval signal before reranking.
    retrieval_score: float | None = None


# ============================================================================
# Collection naming
# ============================================================================


def collection_name_for_session(session_id: str) -> str:
    """
    Generate a deterministic and safe Chroma collection name per session.
    """

    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty.")

    digest = hashlib.sha256(
        session_id.strip().encode("utf-8")
    ).hexdigest()[:16]

    return f"{settings.collection_prefix}_{digest}"


# ============================================================================
# Vector store
# ============================================================================


def build_vector_store(
    chunks: list[LeafletChunk],
    session_id: str,
) -> Chroma:
    """
    Build or replace the Chroma collection for a leaflet session.
    """

    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks.")

    collection_name = collection_name_for_session(session_id)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(VECTOR_DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    # Prevent duplicate chunks when the same session is re-ingested.
    try:
        existing = store.get()

        if existing and existing.get("ids"):
            store.delete(ids=existing["ids"])

    except Exception as exc:
        logger.warning(
            "Could not inspect existing Chroma collection %s: %s",
            collection_name,
            exc,
        )

    documents: list[Document] = []
    ids: list[str] = []

    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            continue

        chunk_id = chunk.chunk_id.strip()

        if not chunk_id:
            chunk_id = _stable_chunk_id(
                chunk.text,
                chunk.page_number,
                chunk.source_file,
            )

        documents.append(
            Document(
                page_content=chunk.text.strip(),
                metadata={
                    "page_number": chunk.page_number,
                    "section": chunk.section or "General",
                    "source_file": chunk.source_file or "leaflet",
                    "chunk_id": chunk_id,
                },
            )
        )

        ids.append(chunk_id)

    if not documents:
        raise ValueError(
            "No valid text chunks were available for indexing."
        )

    store.add_documents(
        documents=documents,
        ids=ids,
    )

    logger.info(
        "Built vector store: collection=%s chunks=%d",
        collection_name,
        len(documents),
    )

    # A new corpus means previous BM25 caches are no longer relevant.
    _clear_bm25_cache()

    return store


def load_vector_store(session_id: str) -> Chroma:
    """
    Load an existing Chroma collection.
    """

    return Chroma(
        collection_name=collection_name_for_session(session_id),
        embedding_function=get_embedding_function(),
        persist_directory=str(VECTOR_DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


# ============================================================================
# Tokenization
# ============================================================================


def tokenize(text: str) -> list[str]:
    """
    Lightweight tokenizer for multilingual BM25 retrieval.
    """

    if not text:
        return []

    return re.findall(
        r"\b[\w'-]+\b",
        text.lower(),
        flags=re.UNICODE,
    )


# ============================================================================
# Generic chunk filter
# ============================================================================


def is_generic(text: str) -> bool:
    """
    Identify chunks that contain only obvious application boilerplate.

    IMPORTANT:
    We intentionally do not filter normal medical leaflet content just
    because it contains phrases such as "consult your doctor".
    """

    if not text or not text.strip():
        return True

    normalized = " ".join(
        text.lower().split()
    )

    # Only filter when the chunk is extremely short and consists mostly
    # of application-level boilerplate.
    generic_phrases = (
        "this is not medical advice",
        "always consult your doctor",
        "consult your doctor",
    )

    if len(normalized) > 180:
        return False

    return any(
        phrase in normalized
        for phrase in generic_phrases
    )


# ============================================================================
# Stable IDs
# ============================================================================


def _stable_chunk_id(
    text: str,
    page_number: int,
    source_file: str,
) -> str:
    """
    Generate a deterministic chunk ID when the parser did not provide one.
    """

    payload = (
        f"{source_file}|"
        f"{page_number}|"
        f"{text.strip()}"
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:24]


# ============================================================================
# Corpus extraction
# ============================================================================


def _get_corpus(
    store: Chroma,
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """
    Extract all stored chunks from Chroma.

    Returns:
        texts
        metadata
        ids
    """

    data = store.get()

    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    ids = data.get("ids") or []

    valid_texts: list[str] = []
    valid_metadata: list[dict[str, Any]] = []
    valid_ids: list[str] = []

    for index, text in enumerate(documents):

        if not text or not text.strip():
            continue

        metadata = (
            dict(metadatas[index])
            if index < len(metadatas)
            and metadatas[index]
            else {}
        )

        chunk_id = (
            ids[index]
            if index < len(ids)
            else metadata.get("chunk_id", "")
        )

        if not chunk_id:
            chunk_id = _stable_chunk_id(
                text,
                int(metadata.get("page_number", 0)),
                str(metadata.get("source_file", "leaflet")),
            )

        metadata.setdefault(
            "chunk_id",
            chunk_id,
        )

        valid_texts.append(text)
        valid_metadata.append(metadata)
        valid_ids.append(str(chunk_id))

    return (
        valid_texts,
        valid_metadata,
        valid_ids,
    )


# ============================================================================
# BM25 cache
# ============================================================================


_BM25_CACHE: dict[
    str,
    tuple[
        BM25Okapi,
        list[str],
        list[dict[str, Any]],
        list[str],
    ],
] = {}


def _corpus_cache_key(
    texts: list[str],
    ids: list[str],
) -> str:
    """
    Generate a deterministic cache key for the current corpus.
    """

    digest = hashlib.sha256()

    for chunk_id, text in zip(ids, texts):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"|")
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")

    return digest.hexdigest()


def _clear_bm25_cache() -> None:
    """Clear cached BM25 indexes."""

    _BM25_CACHE.clear()


def _get_bm25_index(
    texts: list[str],
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> BM25Okapi | None:
    """
    Return a cached BM25 index for the current corpus.
    """

    if not texts:
        return None

    tokenized_corpus = [
        tokenize(text)
        for text in texts
    ]

    if not any(tokenized_corpus):
        return None

    cache_key = _corpus_cache_key(
        texts,
        ids,
    )

    cached = _BM25_CACHE.get(cache_key)

    if cached is not None:
        return cached[0]

    bm25 = BM25Okapi(
        tokenized_corpus
    )

    _BM25_CACHE[cache_key] = (
        bm25,
        texts,
        metadatas,
        ids,
    )

    # Keep the in-memory cache bounded.
    if len(_BM25_CACHE) > 4:
        oldest_key = next(
            iter(_BM25_CACHE)
        )

        if oldest_key != cache_key:
            _BM25_CACHE.pop(
                oldest_key,
                None,
            )

    return bm25


# ============================================================================
# Vector retrieval
# ============================================================================


def _vector_search(
    store: Chroma,
    question: str,
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Retrieve candidates using semantic vector search.

    Returns:
        List of (Document, relevance_score).
    """

    if top_k <= 0:
        return []

    try:
        results = (
            store.similarity_search_with_relevance_scores(
                question,
                k=top_k,
            )
        )

    except Exception as exc:

        logger.debug(
            "Relevance-score retrieval failed; "
            "falling back to raw distance: %s",
            exc,
        )

        raw_results = (
            store.similarity_search_with_score(
                question,
                k=top_k,
            )
        )

        results = [
            (
                document,
                _distance_to_score(
                    distance
                ),
            )
            for document, distance in raw_results
        ]

    normalized: list[
        tuple[Document, float]
    ] = []

    for document, score in results:

        normalized_score = max(
            0.0,
            min(
                1.0,
                float(score),
            ),
        )

        normalized.append(
            (
                document,
                normalized_score,
            )
        )

    return normalized


def _distance_to_score(
    distance: float,
) -> float:
    """
    Convert a distance-like score into a bounded 0..1 value.

    This is only a fallback ranking signal, not a probability.
    """

    try:
        distance = float(distance)
    except (TypeError, ValueError):
        return 0.0

    return max(
        0.0,
        min(
            1.0,
            1.0 - distance,
        ),
    )


# ============================================================================
# BM25 retrieval
# ============================================================================


def _bm25_search(
    store: Chroma,
    question: str,
    top_k: int,
) -> list[tuple[Document, float]]:
    """
    Retrieve candidates using BM25 keyword matching.
    """

    if top_k <= 0:
        return []

    texts, metadatas, ids = _get_corpus(
        store
    )

    if not texts:
        return []

    query_tokens = tokenize(
        question
    )

    if not query_tokens:
        return []

    bm25 = _get_bm25_index(
        texts,
        metadatas,
        ids,
    )

    if bm25 is None:
        return []

    scores = bm25.get_scores(
        query_tokens
    )

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    selected = ranked_indices[:top_k]

    if not selected:
        return []

    selected_scores = [
        float(scores[index])
        for index in selected
    ]

    max_score = max(
        selected_scores,
        default=0.0,
    )

    results: list[
        tuple[Document, float]
    ] = []

    for index, raw_score in zip(
        selected,
        selected_scores,
    ):

        normalized_score = (
            raw_score / max_score
            if max_score > 0
            else 0.0
        )

        metadata = dict(
            metadatas[index]
        )

        metadata.setdefault(
            "chunk_id",
            ids[index],
        )

        document = Document(
            page_content=texts[index],
            metadata=metadata,
        )

        results.append(
            (
                document,
                normalized_score,
            )
        )

    return results


# ============================================================================
# Reciprocal Rank Fusion
# ============================================================================


def _rrf_fusion(
    ranked_lists: list[
        tuple[
            str,
            list[tuple[Document, float]],
        ]
    ],
    top_k: int,
    rrf_k: int = 60,
) -> list[tuple[Document, float]]:
    """
    Combine ranked retrieval lists using weighted RRF.

    Each list is represented as:
        ("vector", results)
        ("bm25", results)

    Weighting allows settings.vector_weight and settings.bm25_weight
    to influence the final hybrid ranking.
    """

    if top_k <= 0:
        return []

    fused_scores: dict[str, float] = {}
    documents: dict[str, Document] = {}

    weights = {
        "vector": max(
            0.0,
            float(
                settings.vector_weight
            ),
        ),
        "bm25": max(
            0.0,
            float(
                settings.bm25_weight
            ),
        ),
    }

    for method, ranked_list in ranked_lists:

        weight = weights.get(
            method,
            1.0,
        )

        if weight <= 0:
            continue

        for rank, (
            document,
            _score,
        ) in enumerate(
            ranked_list,
            start=1,
        ):

            chunk_id = _document_id(
                document
            )

            fused_scores[chunk_id] = (
                fused_scores.get(
                    chunk_id,
                    0.0,
                )
                + weight
                * (
                    1.0
                    / (
                        rrf_k + rank
                    )
                )
            )

            documents[chunk_id] = document

    ranked = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        (
            documents[chunk_id],
            score,
        )
        for chunk_id, score in ranked[:top_k]
    ]


# ============================================================================
# Reranker
# ============================================================================


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Load and cache the configured Cross-Encoder reranker.
    """

    if not settings.reranker_enabled:
        raise RuntimeError(
            "Reranker is disabled in settings."
        )

    logger.info(
        "Loading reranker model: %s",
        settings.reranker_model_name,
    )

    return CrossEncoder(
        settings.reranker_model_name,
        max_length=512,
    )


def _rerank(
    question: str,
    candidates: list[
        tuple[Document, float]
    ],
    top_k: int,
) -> list[
    tuple[Document, float, float]
]:
    """
    Rerank retrieved candidates.

    Returns:
        (document, original_retrieval_score, rerank_score)
    """

    if not candidates or top_k <= 0:
        return []

    if not settings.reranker_enabled:
        return [
            (
                document,
                original_score,
                original_score,
            )
            for document, original_score
            in candidates[:top_k]
        ]

    pairs = [
        (
            question,
            document.page_content,
        )
        for document, _score in candidates
    ]

    scores = get_reranker().predict(
        pairs,
        show_progress_bar=False,
    )

    reranked = [
        (
            document,
            original_score,
            float(rerank_score),
        )
        for (
            document,
            original_score,
        ), rerank_score in zip(
            candidates,
            scores,
        )
    ]

    reranked.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return reranked[:top_k]


# ============================================================================
# Document helpers
# ============================================================================


def _document_id(
    document: Document,
) -> str:
    """
    Get a stable identifier for a Document.
    """

    metadata = document.metadata or {}

    chunk_id = metadata.get(
        "chunk_id"
    )

    if chunk_id:
        return str(chunk_id)

    return hashlib.sha256(
        document.page_content.encode(
            "utf-8"
        )
    ).hexdigest()


def _to_retrieved_chunk(
    document: Document,
    confidence: float,
    method: str,
    rerank_score: float | None = None,
    retrieval_score: float | None = None,
) -> RetrievedChunk:
    """
    Convert a LangChain Document to MediLeaf's public result type.
    """

    metadata = document.metadata or {}

    try:
        page_number = int(
            metadata.get(
                "page_number",
                0,
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        page_number = 0

    return RetrievedChunk(
        text=document.page_content,
        page_number=page_number,
        section=str(
            metadata.get(
                "section",
                "General",
            )
        ),
        source_file=str(
            metadata.get(
                "source_file",
                "leaflet",
            )
        ),
        confidence=round(
            max(
                0.0,
                min(
                    100.0,
                    confidence,
                ),
            ),
            1,
        ),
        chunk_id=str(
            metadata.get(
                "chunk_id",
                "",
            )
        ),
        retrieval_method=method,
        rerank_score=(
            round(
                rerank_score,
                4,
            )
            if rerank_score is not None
            else None
        ),
        retrieval_score=(
            round(
                retrieval_score,
                4,
            )
            if retrieval_score is not None
            else None
        ),
    )


# ============================================================================
# Main retrieval function
# ============================================================================


def retrieve(
    store: Chroma,
    question: str,
    top_k: int | None = None,
    mode: str | None = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant leaflet chunks.

    Supported modes:

        vector
            Semantic vector retrieval only.

        bm25
            Keyword retrieval only.

        hybrid
            BM25 + vector retrieval using weighted RRF.

        hybrid_rerank
            BM25 + vector → RRF → Cross-Encoder reranking.
    """

    if not question or not question.strip():
        return []

    question = question.strip()

    final_top_k = (
        top_k
        if top_k is not None
        else settings.top_k
    )

    final_top_k = max(
        1,
        int(final_top_k),
    )

    mode = (
        mode
        or settings.retrieval_mode
        or "hybrid_rerank"
    ).lower()

    # -----------------------------------------------------------------------
    # Retrieve candidates
    # -----------------------------------------------------------------------

    vector_candidates = _vector_search(
        store,
        question,
        max(
            final_top_k,
            settings.vector_top_k,
        ),
    )

    bm25_candidates = _bm25_search(
        store,
        question,
        max(
            final_top_k,
            settings.bm25_top_k,
        ),
    )

    # -----------------------------------------------------------------------
    # Vector only
    # -----------------------------------------------------------------------

    if mode == "vector":

        retrieved = []

        for document, score in (
            vector_candidates[:final_top_k]
        ):

            if is_generic(
                document.page_content
            ):
                continue

            confidence = score * 100.0

            if confidence < (
                settings.min_relevance_score
                * 100.0
            ):
                continue

            retrieved.append(
                _to_retrieved_chunk(
                    document=document,
                    confidence=confidence,
                    method="vector",
                    retrieval_score=score,
                )
            )

    # -----------------------------------------------------------------------
    # BM25 only
    # -----------------------------------------------------------------------

    elif mode == "bm25":

        retrieved = []

        for document, score in (
            bm25_candidates[:final_top_k]
        ):

            if is_generic(
                document.page_content
            ):
                continue

            retrieved.append(
                _to_retrieved_chunk(
                    document=document,
                    confidence=score * 100.0,
                    method="bm25",
                    retrieval_score=score,
                )
            )

    # -----------------------------------------------------------------------
    # Hybrid / Hybrid + Rerank
    # -----------------------------------------------------------------------

    elif mode in {
        "hybrid",
        "hybrid_rerank",
    }:

        fused = _rrf_fusion(
            [
                (
                    "vector",
                    vector_candidates,
                ),
                (
                    "bm25",
                    bm25_candidates,
                ),
            ],
            top_k=max(
                final_top_k,
                settings.rerank_top_k,
            ),
        )

        if mode == "hybrid_rerank":

            reranked = _rerank(
                question,
                fused,
                settings.rerank_top_k,
            )

            retrieved = []

            for (
                document,
                original_score,
                rerank_score,
            ) in reranked:

                if is_generic(
                    document.page_content
                ):
                    continue

                rerank_confidence = (
                    _reranker_confidence(
                        rerank_score
                    )
                )

                retrieval_confidence = (
                    _rrf_confidence(
                        original_score
                    )
                )

                # Combine reranker relevance with the original
                # hybrid retrieval signal.
                final_confidence = (
                    0.70
                    * rerank_confidence
                    + 0.30
                    * retrieval_confidence
                )

                retrieved.append(
                    _to_retrieved_chunk(
                        document=document,
                        confidence=final_confidence,
                        method="hybrid_rerank",
                        rerank_score=rerank_score,
                        retrieval_score=original_score,
                    )
                )

        else:

            retrieved = []

            for document, score in (
                fused[:final_top_k]
            ):

                if is_generic(
                    document.page_content
                ):
                    continue

                retrieved.append(
                    _to_retrieved_chunk(
                        document=document,
                        confidence=_rrf_confidence(
                            score
                        ),
                        method="hybrid",
                        retrieval_score=score,
                    )
                )

    else:

        logger.warning(
            "Unknown retrieval mode '%s'. "
            "Falling back to hybrid_rerank.",
            mode,
        )

        return retrieve(
            store=store,
            question=question,
            top_k=final_top_k,
        )

    # -----------------------------------------------------------------------
    # Final deduplication
    # -----------------------------------------------------------------------

    retrieved = _deduplicate_results(
        retrieved
    )

    retrieved.sort(
        key=lambda chunk: chunk.confidence,
        reverse=True,
    )

    final_results = retrieved[
        :final_top_k
    ]

    # -----------------------------------------------------------------------
    # Structured logging
    # -----------------------------------------------------------------------

    logger.info(
        "Retrieval completed | mode=%s | "
        "question=%r | results=%d",
        mode,
        question,
        len(final_results),
    )

    for index, chunk in enumerate(
        final_results,
        start=1,
    ):
        logger.debug(
            "Result #%d | score=%.1f | "
            "method=%s | page=%s | "
            "section=%s | chunk=%s",
            index,
            chunk.confidence,
            chunk.retrieval_method,
            chunk.page_number,
            chunk.section,
            chunk.chunk_id,
        )

    return final_results


# ============================================================================
# Score conversion
# ============================================================================


def _rrf_confidence(
    score: float,
) -> float:
    """
    Convert an RRF score into a display-friendly ranking signal.

    IMPORTANT:
    This is NOT a probability.
    """

    return min(
        100.0,
        max(
            0.0,
            float(score) * 1000.0,
        ),
    )


def _reranker_confidence(
    score: float,
) -> float:
    """
    Convert Cross-Encoder score into a bounded ranking signal.

    IMPORTANT:
    This is NOT a calibrated probability.
    """

    score = max(
        -20.0,
        min(
            20.0,
            float(score),
        ),
    )

    probability = (
        1.0
        / (
            1.0
            + math.exp(-score)
        )
    )

    return probability * 100.0


# ============================================================================
# Deduplication
# ============================================================================


def _deduplicate_results(
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Remove duplicate chunks while preserving the highest-ranked result.
    """

    seen: set[str] = set()
    unique: list[RetrievedChunk] = []

    for chunk in chunks:

        identifier = (
            chunk.chunk_id
            or hashlib.sha256(
                chunk.text.encode(
                    "utf-8"
                )
            ).hexdigest()
        )

        if identifier in seen:
            continue

        seen.add(identifier)
        unique.append(chunk)

    return unique


# ============================================================================
# Context validation
# ============================================================================


def has_sufficient_context(
    chunks: list[RetrievedChunk],
) -> bool:
    """
    Decide whether retrieved context is strong enough for generation.
    """

    if not chunks:
        return False

    if len(chunks) < (
        settings.min_context_chunks
    ):
        return False

    best_confidence = max(
        chunk.confidence
        for chunk in chunks
    )

    return (
        best_confidence
        >= settings.min_relevance_score
        * 100.0
    )


def overall_confidence(
    chunks: list[RetrievedChunk],
) -> float:
    """
    Return the strongest retrieval confidence.
    """

    if not chunks:
        return 0.0

    return round(
        max(
            chunk.confidence
            for chunk in chunks
        ),
        1,
    )