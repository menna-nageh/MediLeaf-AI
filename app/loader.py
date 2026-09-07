"""
loader.py
---------
PDF loading utilities for MediLeaf AI.

Responsible for:
- Safe PDF opening
- Page-by-page text extraction
- Preserving page boundaries
- Basic PDF validation
- Returning structured page data for downstream parsing
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class PDFPage:
    """
    Represents the extracted content of a single PDF page.
    """

    page_number: int
    text: str


# ============================================================
# PDF VALIDATION
# ============================================================

def validate_pdf(file_path: str | Path) -> None:
    """
    Validate that the provided file exists and is a readable PDF.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")


# ============================================================
# PAGE EXTRACTION
# ============================================================

def extract_pages_from_pdf(file_path: str | Path) -> list[PDFPage]:
    """
    Extract text from a PDF page by page.

    Keeping page boundaries is important for:
    - Retrieval metadata
    - Source citations
    - Debugging
    - Evaluation
    """

    validate_pdf(file_path)

    pages: list[PDFPage] = []

    try:
        with fitz.open(file_path) as doc:

            if len(doc) == 0:
                raise ValueError("The PDF contains no pages.")

            for page_index, page in enumerate(doc):
                text = page.get_text("text") or ""

                text = text.strip()

                if not text:
                    continue

                pages.append(
                    PDFPage(
                        page_number=page_index + 1,
                        text=text,
                    )
                )

    except fitz.FileDataError as exc:
        raise ValueError(
            "The PDF could not be opened. "
            "It may be corrupted, encrypted, or invalid."
        ) from exc

    return pages


# ============================================================
# FULL TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(file_path: str | Path) -> str:
    """
    Extract the complete text from a PDF while preserving page boundaries.

    Page separators make the resulting text easier to debug and prevent
    content from adjacent pages from being merged invisibly.
    """

    pages = extract_pages_from_pdf(file_path)

    if not pages:
        return ""

    page_blocks = []

    for page in pages:
        page_blocks.append(
            f"\n--- PAGE {page.page_number} ---\n"
            f"{page.text}"
        )

    return "\n".join(page_blocks).strip()


# ============================================================
# PDF METADATA
# ============================================================

def get_pdf_metadata(file_path: str | Path) -> dict[str, str]:
    """
    Return basic PDF metadata useful for logging and ingestion.
    """

    validate_pdf(file_path)

    try:
        with fitz.open(file_path) as doc:
            metadata = doc.metadata or {}

            return {
                "title": metadata.get("title", "") or "",
                "author": metadata.get("author", "") or "",
                "subject": metadata.get("subject", "") or "",
                "creator": metadata.get("creator", "") or "",
                "producer": metadata.get("producer", "") or "",
                "page_count": str(len(doc)),
            }

    except fitz.FileDataError as exc:
        raise ValueError(
            "Unable to read PDF metadata."
        ) from exc


# ============================================================
# SIMPLE PDF INFO
# ============================================================

def get_pdf_page_count(file_path: str | Path) -> int:
    """
    Return the number of pages in a PDF.
    """

    validate_pdf(file_path)

    try:
        with fitz.open(file_path) as doc:
            return len(doc)

    except fitz.FileDataError as exc:
        raise ValueError(
            "Unable to read PDF."
        ) from exc