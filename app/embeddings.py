"""
embeddings.py
-------------
Wraps the HuggingFace embedding model used to turn leaflet text chunks
(and user questions) into vectors.

The embedding model is loaded once and cached because loading the
SentenceTransformer model is expensive.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model.

    Uses BAAI/bge-m3:
    - multilingual support
    - strong semantic retrieval
    - suitable for medical leaflet RAG
    """

    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        },
    )


def embed_query(text: str) -> list[float]:
    """
    Embed a user question/query.
    """
    return get_embedding_function().embed_query(text)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed leaflet chunks/documents.
    """
    return get_embedding_function().embed_documents(texts)