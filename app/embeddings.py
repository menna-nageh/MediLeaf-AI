"""
embeddings.py
-------------
Embedding layer for MediLeaf AI.

This module provides a cached HuggingFace embedding model used for:
- Document/chunk embeddings
- Query embeddings
- Vector retrieval

The actual model is configured through environment variables and
centralised in app.config.Settings.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import settings


@lru_cache(maxsize=1)
def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Return a cached HuggingFace embedding model.

    The model name is controlled by EMBEDDING_MODEL_NAME.

    Default:
        intfloat/multilingual-e5-base

    The model is cached because loading a SentenceTransformer model
    is relatively expensive.
    """

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )


def embed_query(text: str) -> list[float]:
    """
    Convert a user query into a normalized embedding vector.
    """

    if not text or not text.strip():
        raise ValueError("Query text cannot be empty.")

    return get_embedding_function().embed_query(text.strip())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Convert document chunks into normalized embedding vectors.
    """

    if not texts:
        return []

    cleaned_texts = [
        text.strip()
        for text in texts
        if text and text.strip()
    ]

    if not cleaned_texts:
        return []

    return get_embedding_function().embed_documents(cleaned_texts)


def get_embedding_model_name() -> str:
    """
    Return the currently configured embedding model name.

    Useful for logging, debugging, and evaluation reports.
    """

    return settings.embedding_model_name