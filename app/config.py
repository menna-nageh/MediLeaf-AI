"""
config.py
---------
Centralised configuration for MediLeaf AI.

All tunable application parameters are loaded from environment
variables so the system can run locally, in Docker, or in the cloud
without hard-coded configuration values.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# BASE PATHS
# ============================================================

BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
PDF_DIR: Path = DATA_DIR / "pdfs"
VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
LOGS_DIR: Path = BASE_DIR / "logs"
STYLES_DIR: Path = BASE_DIR / "styles"
ASSETS_DIR: Path = BASE_DIR / "assets"

for _directory in (
    DATA_DIR,
    PDF_DIR,
    VECTOR_DB_DIR,
    LOGS_DIR,
    STYLES_DIR,
    ASSETS_DIR,
):
    _directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

@dataclass(frozen=True)
class Settings:
    """
    Immutable application configuration.

    All values can be overridden through environment variables.
    """

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    app_name: str = "MediLeaf AI"

    app_tagline: str = (
        "Understand Any Medicine Leaflet in Seconds"
    )

    log_level: str = field(
        default_factory=lambda: os.getenv(
            "LOG_LEVEL",
            "INFO",
        ).upper()
    )

    # --------------------------------------------------------
    # LLM
    # --------------------------------------------------------

    google_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_API_KEY",
            "",
        ).strip()
    )

    llm_model_name: str = field(
        default_factory=lambda: os.getenv(
            "LLM_MODEL_NAME",
            "gemini-1.5-flash",
        ).strip()
    )

    llm_temperature: float = field(
        default_factory=lambda: float(
            os.getenv(
                "LLM_TEMPERATURE",
                "0.1",
            )
        )
    )

    llm_max_output_tokens: int = field(
        default_factory=lambda: int(
            os.getenv(
                "LLM_MAX_OUTPUT_TOKENS",
                "1024",
            )
        )
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    embedding_model_name: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL_NAME",
            "intfloat/multilingual-e5-base",
        ).strip()
    )

    # --------------------------------------------------------
    # Chunking
    # --------------------------------------------------------

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv(
                "CHUNK_SIZE",
                "800",
            )
        )
    )

    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv(
                "CHUNK_OVERLAP",
                "150",
            )
        )
    )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_mode: str = field(
        default_factory=lambda: os.getenv(
            "RETRIEVAL_MODE",
            "hybrid_rerank",
        ).strip().lower()
    )

    top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "TOP_K",
                "5",
            )
        )
    )

    vector_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "VECTOR_TOP_K",
                "10",
            )
        )
    )

    bm25_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "BM25_TOP_K",
                "10",
            )
        )
    )

    rerank_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "RERANK_TOP_K",
                "5",
            )
        )
    )

    # --------------------------------------------------------
    # Hybrid Retrieval
    # --------------------------------------------------------

    vector_weight: float = field(
        default_factory=lambda: float(
            os.getenv(
                "VECTOR_WEIGHT",
                "0.5",
            )
        )
    )

    bm25_weight: float = field(
        default_factory=lambda: float(
            os.getenv(
                "BM25_WEIGHT",
                "0.5",
            )
        )
    )

    # --------------------------------------------------------
    # Reranking
    # --------------------------------------------------------

    reranker_model_name: str = field(
        default_factory=lambda: os.getenv(
            "RERANKER_MODEL_NAME",
            "BAAI/bge-reranker-base",
        ).strip()
    )

    reranker_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "RERANKER_ENABLED",
            "true",
        ).strip().lower() == "true"
    )

    # --------------------------------------------------------
    # Relevance / Grounding
    # --------------------------------------------------------

    min_relevance_score: float = field(
        default_factory=lambda: float(
            os.getenv(
                "MIN_RELEVANCE_SCORE",
                "0.15",
            )
        )
    )

    min_context_chunks: int = field(
        default_factory=lambda: int(
            os.getenv(
                "MIN_CONTEXT_CHUNKS",
                "1",
            )
        )
    )

    citation_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CITATION_ENABLED",
            "true",
        ).strip().lower() == "true"
    )

    grounding_check_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "GROUNDING_CHECK_ENABLED",
            "true",
        ).strip().lower() == "true"
    )

    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    vector_db_dir: Path = VECTOR_DB_DIR

    collection_prefix: str = field(
        default_factory=lambda: os.getenv(
            "COLLECTION_PREFIX",
            "medileaf",
        ).strip()
    )

    reset_chroma_on_ingest: bool = field(
        default_factory=lambda: os.getenv(
            "RESET_CHROMA_ON_INGEST",
            "false",
        ).strip().lower() == "true"
    )

    # --------------------------------------------------------
    # Conversation Memory
    # --------------------------------------------------------

    memory_window: int = field(
        default_factory=lambda: int(
            os.getenv(
                "MEMORY_WINDOW",
                "3",
            )
        )
    )

    # --------------------------------------------------------
    # API
    # --------------------------------------------------------

    api_host: str = field(
        default_factory=lambda: os.getenv(
            "API_HOST",
            "0.0.0.0",
        ).strip()
    )

    api_port: int = field(
        default_factory=lambda: int(
            os.getenv(
                "API_PORT",
                "8000",
            )
        )
    )

    api_key: str = field(
        default_factory=lambda: os.getenv(
            "MEDILEAF_API_KEY",
            "",
        ).strip()
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    evaluation_top_k: int = field(
        default_factory=lambda: int(
            os.getenv(
                "EVALUATION_TOP_K",
                "5",
            )
        )
    )

    # --------------------------------------------------------
    # User-facing messages
    # --------------------------------------------------------

    disclaimer: str = (
        "This information is generated from the uploaded leaflet only. "
        "It is not medical advice, a diagnosis, or a prescription. "
        "Always consult a doctor or pharmacist before making any medical "
        "decision."
    )

    insufficient_info_message: str = (
        "The uploaded leaflet does not contain enough information "
        "to answer this question."
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    def validate(self) -> list[str]:
        """
        Return human-readable configuration problems.
        """

        problems: list[str] = []

        # ----------------------------------------------------
        # API / LLM
        # ----------------------------------------------------

        if not self.google_api_key:
            problems.append(
                "GOOGLE_API_KEY is not set. Add it to .env before "
                "starting the application."
            )

        if not self.llm_model_name:
            problems.append(
                "LLM_MODEL_NAME cannot be empty."
            )

        if not 0 <= self.llm_temperature <= 2:
            problems.append(
                "LLM_TEMPERATURE must be between 0 and 2."
            )

        if self.llm_max_output_tokens < 1:
            problems.append(
                "LLM_MAX_OUTPUT_TOKENS must be greater than zero."
            )

        if not self.embedding_model_name:
            problems.append(
                "EMBEDDING_MODEL_NAME cannot be empty."
            )

        # ----------------------------------------------------
        # Chunking
        # ----------------------------------------------------

        if self.chunk_size <= 0:
            problems.append(
                "CHUNK_SIZE must be greater than zero."
            )

        if self.chunk_overlap < 0:
            problems.append(
                "CHUNK_OVERLAP cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:
            problems.append(
                "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
            )

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        if self.top_k < 1:
            problems.append(
                "TOP_K must be at least 1."
            )

        if self.vector_top_k < 1:
            problems.append(
                "VECTOR_TOP_K must be at least 1."
            )

        if self.bm25_top_k < 1:
            problems.append(
                "BM25_TOP_K must be at least 1."
            )

        if self.rerank_top_k < 1:
            problems.append(
                "RERANK_TOP_K must be at least 1."
            )

        allowed_modes = {
            "vector",
            "bm25",
            "hybrid",
            "hybrid_rerank",
        }

        if self.retrieval_mode not in allowed_modes:
            problems.append(
                "RETRIEVAL_MODE must be one of: "
                + ", ".join(sorted(allowed_modes))
                + "."
            )

        # ----------------------------------------------------
        # Hybrid weights
        # ----------------------------------------------------

        if not 0 <= self.vector_weight <= 1:
            problems.append(
                "VECTOR_WEIGHT must be between 0 and 1."
            )

        if not 0 <= self.bm25_weight <= 1:
            problems.append(
                "BM25_WEIGHT must be between 0 and 1."
            )

        if abs(
            (self.vector_weight + self.bm25_weight) - 1.0
        ) > 1e-6:
            problems.append(
                "VECTOR_WEIGHT + BM25_WEIGHT must equal 1.0."
            )

        # ----------------------------------------------------
        # Relevance / Grounding
        # ----------------------------------------------------

        if not 0 <= self.min_relevance_score <= 1:
            problems.append(
                "MIN_RELEVANCE_SCORE must be between 0 and 1."
            )

        if self.min_context_chunks < 1:
            problems.append(
                "MIN_CONTEXT_CHUNKS must be at least 1."
            )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        if self.memory_window < 0:
            problems.append(
                "MEMORY_WINDOW cannot be negative."
            )

        # ----------------------------------------------------
        # API
        # ----------------------------------------------------

        if self.api_port < 1 or self.api_port > 65535:
            problems.append(
                "API_PORT must be between 1 and 65535."
            )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if self.evaluation_top_k < 1:
            problems.append(
                "EVALUATION_TOP_K must be at least 1."
            )

        return problems


# ============================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================

settings = Settings()


# ============================================================
# EMERGENCY KEYWORDS
# ============================================================

EMERGENCY_KEYWORDS: tuple[str, ...] = (
    # English
    "overdose",
    "took too much",
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
    "heavy bleeding",
    "seizure",
    "convulsion",
    "swelling of the face",
    "swelling of the throat",
    "can't breathe",
    "cannot breathe",
    "suicidal",
    "poisoning",

    # Arabic
    "جرعة زائدة",
    "تناولت جرعة زائدة",
    "أخذت جرعة زائدة",
    "حساسية شديدة",
    "رد فعل تحسسي شديد",
    "صعوبة في التنفس",
    "ضيق التنفس",
    "فقدان الوعي",
    "فاقد الوعي",
    "ألم في الصدر",
    "نزيف شديد",
    "تشنجات",
    "تورم الوجه",
    "تورم الحلق",
    "لا أستطيع التنفس",
    "تسمم",
)


# ============================================================
# QUICK QUESTIONS
# ============================================================

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