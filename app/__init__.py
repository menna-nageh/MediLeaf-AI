"""
MediLeaf AI - application package.

This package contains the core RAG pipeline modules:
    config          -> environment / app configuration
    parser          -> PDF loading and text cleaning
    embeddings      -> embedding model wrapper (HuggingFace)
    retriever       -> ChromaDB vector store + semantic retrieval
    prompt_builder  -> prompt templates for answers and summaries
    llm             -> Google Gemini 2.5 Flash wrapper
    memory          -> short-term conversation memory
    emergency       -> emergency keyword detection
"""

__version__ = "1.0.0"
