"""
parser.py
---------
PDF ingestion, cleaning, section detection, and structure-aware chunking
for MediLeaf AI.

The parser preserves page-level provenance so every retrieved chunk can
later be cited and evaluated.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


# ============================================================================
# Exceptions
# ============================================================================

class PDFParsingError(Exception):
    """Raised when a PDF cannot be parsed or contains no usable text."""


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class PageText:
    """Extracted text belonging to one PDF page."""

    page_number: int
    text: str


@dataclass
class LeafletChunk:
    """
    A retrieval-ready chunk with provenance metadata.

    Keeping page number, section, and source file attached to every chunk
    is essential for grounded answers and citations.
    """

    chunk_id: str
    text: str
    page_number: int
    source_file: str
    section: str = "General"


# ============================================================================
# PDF extraction
# ============================================================================

def extract_pages(
    pdf_source: str | Path | BinaryIO,
    source_name: str = "leaflet.pdf",
) -> list[PageText]:
    """
    Extract text page-by-page from a PDF.

    Args:
        pdf_source:
            File path, pathlib.Path, or binary file-like object.

        source_name:
            Human-readable source name used in error messages/logging.

    Returns:
        A list of non-empty PageText objects.

    Raises:
        PDFParsingError:
            If the PDF cannot be opened or contains no extractable text.
    """

    try:
        if isinstance(pdf_source, (str, Path)):
            document = fitz.open(pdf_source)
        else:
            pdf_bytes = pdf_source.read()

            if not pdf_bytes:
                raise PDFParsingError(
                    f"Empty PDF source: {source_name}"
                )

            document = fitz.open(
                stream=pdf_bytes,
                filetype="pdf",
            )

    except PDFParsingError:
        raise

    except Exception as exc:
        raise PDFParsingError(
            f"Unable to open PDF '{source_name}'."
        ) from exc

    pages: list[PageText] = []

    try:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)

            text = page.get_text("text")

            cleaned = clean_text(text)

            if cleaned:
                pages.append(
                    PageText(
                        page_number=page_index + 1,
                        text=cleaned,
                    )
                )
    finally:
        document.close()

    if not pages:
        raise PDFParsingError(
            f"No extractable text found in PDF '{source_name}'. "
            "The document may be scanned/image-only."
        )

    return pages


# ============================================================================
# Text cleaning
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving meaningful line boundaries.

    Important:
        We intentionally DO NOT collapse all whitespace into spaces.
        Newlines are useful for detecting sections and creating better chunks.
    """

    if not text:
        return ""

    # Remove null characters.
    text = text.replace("\x00", " ")

    # Normalize different newline styles.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Rejoin words split by PDF line wrapping, then turn single line breaks
    # into spaces while preserving paragraph boundaries.
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Remove common Unicode formatting artifacts.
    text = text.replace("\ufeff", "")
    text = text.replace("\u00ad", "")

    # Remove characters that are usually extraction noise while preserving:
    # Arabic, Latin, numbers, punctuation, and common medical symbols.
    text = re.sub(
        r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
        r"A-Za-z0-9"
        r"\s"
        r".,:;%()\-+/°µμ²³'\"!?&*=<>[\]{}]",
        " ",
        text,
    )

    # Clean spaces around newlines.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    # Avoid excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Avoid repeated horizontal whitespace while preserving newlines.
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ============================================================================
# Section detection
# ============================================================================

_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "Uses",
        re.compile(
            r"\b(uses?|used for|indications?|what is .* used for)\b"
            r"|استعمال|دواعي الاستعمال|يستخدم|الاستخدامات",
            re.IGNORECASE,
        ),
    ),
    (
        "Dosage",
        re.compile(
            r"\b(dosage|dose|how to take|how much|administration)\b"
            r"|الجرعة|طريقة الاستخدام|كيفية الاستخدام",
            re.IGNORECASE,
        ),
    ),
    (
        "Warnings",
        re.compile(
            r"\b(warnings?|precautions?|important information|caution)\b"
            r"|تحذير|تحذيرات|احتياطات|تنبيه",
            re.IGNORECASE,
        ),
    ),
    (
        "Contraindications",
        re.compile(
            r"\b(contraindications?|do not use|should not use)\b"
            r"|موانع الاستعمال|لا تستخدم|يمنع استخدام",
            re.IGNORECASE,
        ),
    ),
    (
        "Side Effects",
        re.compile(
            r"\b(side effects?|adverse effects?|undesirable effects?)\b"
            r"|الآثار الجانبية|أعراض جانبية",
            re.IGNORECASE,
        ),
    ),
    (
        "Interactions",
        re.compile(
            r"\b(drug interactions?|interactions?|other medicines?|medicines?)\b"
            r"|التداخلات الدوائية|التداخلات|أدوية أخرى",
            re.IGNORECASE,
        ),
    ),
    (
        "Pregnancy and Breastfeeding",
        re.compile(
            r"\b(pregnancy|pregnant|breastfeeding|breast-feeding|lactation)\b"
            r"|الحمل|الرضاعة|الرضاعة الطبيعية",
            re.IGNORECASE,
        ),
    ),
    (
        "Children",
        re.compile(
            r"\b(children|child|paediatric|pediatric|infants?)\b"
            r"|الأطفال|طفل|الرضع",
            re.IGNORECASE,
        ),
    ),
    (
        "Storage",
        re.compile(
            r"\b(storage|store|keep|temperature|expiry|expiration)\b"
            r"|التخزين|يحفظ|الحفظ|درجة الحرارة|الصلاحية",
            re.IGNORECASE,
        ),
    ),
    (
        "Overdose",
        re.compile(
            r"\b(overdose|too much|excessive dose)\b"
            r"|الجرعة الزائدة|جرعة زائدة",
            re.IGNORECASE,
        ),
    ),
)


def _guess_section(text: str) -> str:
    """
    Infer a likely leaflet section from the chunk text.

    This is intentionally heuristic. It provides useful metadata for
    retrieval, citations, and debugging without pretending to be a
    perfect document-structure parser.
    """

    for section_name, pattern in _SECTION_PATTERNS:
        if pattern.search(text):
            return section_name

    return "General"


# ============================================================================
# Chunk ID
# ============================================================================

def _make_chunk_id(
    source_name: str,
    page_number: int,
    chunk_index: int,
    text: str,
) -> str:
    """
    Create a deterministic chunk ID.

    Including a short content hash makes IDs stable and collision-resistant
    while keeping them reasonably short.
    """

    content_hash = hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:10]

    safe_source = Path(source_name).stem[:40]

    return (
        f"{safe_source}"
        f"_p{page_number}"
        f"_c{chunk_index}"
        f"_{content_hash}"
    )


# ============================================================================
# Chunking
# ============================================================================

def chunk_pages(
    pages: list[PageText],
    source_name: str,
) -> list[LeafletChunk]:
    """
    Split leaflet pages into retrieval-ready chunks.

    Chunk size and overlap are controlled centrally through Settings.

    Page boundaries are preserved, meaning a chunk never accidentally
    receives the page number of a different page.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "؟ ",
            "؛ ",
            "، ",
            " ",
            "",
        ],
        length_function=len,
        is_separator_regex=False,
    )

    chunks: list[LeafletChunk] = []

    for page in pages:
        cleaned_text = clean_text(page.text)

        if not cleaned_text:
            continue

        pieces = splitter.split_text(cleaned_text)

        page_chunk_index = 0

        for piece in pieces:
            piece = piece.strip()

            # Ignore extremely short/noisy fragments.
            if len(piece) < 30:
                continue

            # Ignore chunks consisting almost entirely of punctuation.
            alphanumeric_count = len(
                re.findall(
                    r"[\w\u0600-\u06FF]",
                    piece,
                    flags=re.UNICODE,
                )
            )

            if alphanumeric_count < 15:
                continue

            page_chunk_index += 1

            section = _guess_section(piece)

            chunk_id = _make_chunk_id(
                source_name=source_name,
                page_number=page.page_number,
                chunk_index=page_chunk_index,
                text=piece,
            )

            chunks.append(
                LeafletChunk(
                    chunk_id=chunk_id,
                    text=piece,
                    page_number=page.page_number,
                    source_file=source_name,
                    section=section,
                )
            )

    if not chunks:
        raise PDFParsingError(
            f"PDF '{source_name}' contained text, but no usable chunks "
            "could be created."
        )

    return chunks


# ============================================================================
# Public parsing API
# ============================================================================

def parse_pdf(
    pdf_source: str | Path | BinaryIO,
    source_name: str = "leaflet.pdf",
) -> list[LeafletChunk]:
    """
    Full PDF ingestion pipeline:

        PDF
         ↓
        page extraction
         ↓
        cleaning
         ↓
        structure-aware chunking
         ↓
        provenance metadata
    """

    pages = extract_pages(
        pdf_source,
        source_name,
    )

    return chunk_pages(
        pages,
        source_name,
    )


def full_text(
    pages: list[PageText],
) -> str:
    """
    Return cleaned full document text while preserving page boundaries.
    """

    return "\n\n".join(
        page.text
        for page in pages
        if page.text
    )