"""
retriever.py
------------
Builds and queries a per-leaflet ChromaDB collection.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.config import VECTOR_DB_DIR, settings
from app.embeddings import get_embedding_function
from app.parser import LeafletChunk


@dataclass
class RetrievedChunk:
    text: str
    page_number: int
    section: str
    source_file: str
    confidence: float  # 0-100


# =========================
# Generic Chunk Filter (light)
# =========================
GENERIC_PATTERNS = [
    "this is not medical advice",
    "consult your doctor",
    "always consult",
]


def is_generic(text: str) -> bool:
    """Light filter — only blocks obvious boilerplate, never real leaflet content."""
    t = text.lower()
    return any(p in t for p in GENERIC_PATTERNS)


# =========================
# Collection Naming
# =========================
def collection_name_for_session(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]
    return f"{settings.collection_prefix}_{digest}"


# =========================
# Build Vector Store
# =========================
def build_vector_store(chunks: list[LeafletChunk], session_id: str) -> Chroma:
    collection_name = collection_name_for_session(session_id)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory=str(VECTOR_DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    existing = store.get()
    if existing and existing.get("ids"):
        store.delete(ids=existing["ids"])

    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "page_number": chunk.page_number,
                "section": chunk.section,
                "source_file": chunk.source_file,
                "chunk_id": chunk.chunk_id,
            },
        )
        for chunk in chunks
    ]

    ids = [chunk.chunk_id for chunk in chunks]

    store.add_documents(documents=documents, ids=ids)

    return store


def load_vector_store(session_id: str) -> Chroma:
    return Chroma(
        collection_name=collection_name_for_session(session_id),
        embedding_function=get_embedding_function(),
        persist_directory=str(VECTOR_DB_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


# =========================
# Retrieve
# =========================
def retrieve(store: Chroma, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    k = top_k or settings.top_k

    # Use similarity search with relevance scores from actual distances
    results_with_scores = store.similarity_search_with_relevance_scores(
        question,
        k=k
    )

    retrieved: list[RetrievedChunk] = []

    for document, relevance_score in results_with_scores:
        text = document.page_content

        # Skip only obvious boilerplate
        if is_generic(text):
            continue

        # Convert relevance score to confidence percentage
        confidence = round(relevance_score * 100, 1)

        # Only include chunks with reasonable relevance
        if confidence < 10:
            continue

        retrieved.append(
            RetrievedChunk(
                text=text,
                page_number=document.metadata.get("page_number", 0),
                section=document.metadata.get("section", "General"),
                source_file=document.metadata.get("source_file", "leaflet"),
                confidence=confidence,
            )
        )

    # Sort by confidence descending
    retrieved.sort(key=lambda x: x.confidence, reverse=True)

    print("\n[DEBUG] QUESTION:", question)
    for c in retrieved:
        print(f"[DEBUG] Score: {c.confidence} | Section: {c.section} | Chunk: {c.text[:80]}...")

    return retrieved[:k]


# =========================
# Context Validation
# =========================
def has_sufficient_context(chunks: list[RetrievedChunk]) -> bool:
    if not chunks:
        return False

    best = max(c.confidence for c in chunks)
    return best >= 30


def overall_confidence(chunks: list[RetrievedChunk]) -> float:
    if not chunks:
        return 0.0
    return max(c.confidence for c in chunks)

