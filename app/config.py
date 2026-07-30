"""
config.py
---------
Centralised configuration for MediLeaf AI.

All tunable parameters (model names, chunking strategy, retrieval depth,
storage paths, etc.) live here so the rest of the application never
hard-codes a "magic value". Values are loaded from environment variables
(via a `.env` file in development) with sensible defaults, which makes the
app portable across local machines, containers and cloud deployments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Load variables from a .env file (if present) into the process environment.
# This must happen before any os.getenv() calls below.
load_dotenv()

# --------------------------------------------------------------------------- #
# Base paths
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
LOGS_DIR: Path = BASE_DIR / "logs"
STYLES_DIR: Path = BASE_DIR / "styles"
ASSETS_DIR: Path = BASE_DIR / "assets"

for _dir in (DATA_DIR, PDF_DIR, VECTOR_DB_DIR, LOGS_DIR, STYLES_DIR, ASSETS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings, populated once at import time."""

    # --- Google Gemini -----------------------------------------------------
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    llm_model_name: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )
    llm_max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1024"))
    )

    # --- Embeddings ----------------------------------------------------------
    # Kept as a separate, swappable setting: the RAG pipeline only ever talks
    # to app.embeddings.get_embedding_function(), so changing this string is
    # enough to switch embedding models later without touching other files.
    embedding_model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "intfloat/multilingual-e5-base")
    )

    # --- Chunking --------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))

    # --- Retrieval -----------------------------------------------------------
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    # Chroma returns a distance (lower = more similar). We convert this to a
    # 0-100% "confidence" score for display; scores below this floor are
    # treated as "not relevant enough" when deciding whether context exists.
    min_relevance_score: float = field(
        default_factory=lambda: float(os.getenv("MIN_RELEVANCE_SCORE", "0.15"))
    )

    # --- Memory ---------------------------------------------------------
    memory_window: int = field(default_factory=lambda: int(os.getenv("MEMORY_WINDOW", "3")))

    # --- Collection naming -------------------------------------------------
    collection_prefix: str = field(
        default_factory=lambda: os.getenv("COLLECTION_PREFIX", "medileaf")
    )

    # --- Misc -----------------------------------------------------------
    app_name: str = "MediLeaf AI"
    app_tagline: str = "Understand Any Medicine Leaflet in Seconds"
    disclaimer: str = (
        "This information is generated from the uploaded leaflet only. "
        "It is not medical advice, a diagnosis, or a prescription. "
        "Always consult a doctor or pharmacist before making any medical decision."
    )
    insufficient_info_message: str = (
        "The uploaded leaflet does not contain enough information to answer this question."
    )

    def validate(self) -> list[str]:
        """Return a list of human-readable configuration problems, if any."""
        problems: list[str] = []
        if not self.google_api_key:
            problems.append(
                "GOOGLE_API_KEY is not set. Add it to a .env file or your environment "
                "before starting the app (see .env.example)."
            )
        if self.chunk_overlap >= self.chunk_size:
            problems.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")
        if self.top_k < 1:
            problems.append("TOP_K must be at least 1.")
        return problems


settings = Settings()


EMERGENCY_KEYWORDS: tuple[str, ...] = (
    "overdose",
    "severe allergy",
    "severe allergic reaction",
    "anaphylaxis",
    "difficulty breathing",
    "trouble breathing",
    "shortness of breath",
    "loss of consciousness",
    "unconscious",
    "unresponsive",
    "chest pain",
    "bleeding",
    "heavy bleeding",
    "seizure",
    "convulsion",
    "swelling of the face",
    "swelling of the throat",
    "can't breathe",
    "cannot breathe",
    "suicidal",
    "poisoning",
)

QUICK_QUESTIONS: tuple[str, ...] = (
    "What is this medicine used for?",
    "Common side effects",
    "How should I take it?",
    "Before or after food?",
    "Missed dose",
    "Overdose",
    "Storage",
    "Pregnancy",
    "Breastfeeding",
    "Children",
    "Warnings",
    "Drug interactions",
)
