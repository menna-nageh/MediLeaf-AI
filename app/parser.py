# """
# parser.py
# ---------
# Responsible for turning an uploaded PDF leaflet into clean, page-tagged text
# chunks that are ready for embedding.

# Pipeline stage: Extract PDF -> Clean Text -> Chunking
# """

# from __future__ import annotations

# import re
# from dataclasses import dataclass
# from pathlib import Path
# from typing import BinaryIO, Union

# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# from app.config import settings


# class PDFParsingError(Exception):
#     """Raised when a PDF cannot be opened or contains no extractable text."""


# @dataclass
# class PageText:
#     """Raw text extracted from a single PDF page."""

#     page_number: int  # 1-indexed, human friendly
#     text: str


# @dataclass
# class LeafletChunk:
#     """A single chunk of leaflet text, ready for embedding."""

#     chunk_id: str
#     text: str
#     page_number: int
#     source_file: str
#     section: str = "General"


# def extract_pages(pdf_source: Union[str, Path, BinaryIO], source_name: str) -> list[PageText]:
#     """
#     Extract raw text from every page of a PDF.

#     Args:
#         pdf_source: A file path, or a file-like object (e.g. Streamlit's
#             UploadedFile), containing PDF bytes.
#         source_name: Human readable name of the source file, used in logs
#             and error messages.

#     Returns:
#         A list of PageText, one entry per non-empty page.

#     Raises:
#         PDFParsingError: if the file cannot be opened or is empty/corrupted.
#     """
#     try:
#         if isinstance(pdf_source, (str, Path)):
#             doc = fitz.open(pdf_source)
#         else:
#             # File-like object (e.g. Streamlit UploadedFile) - read bytes.
#             pdf_bytes = pdf_source.read()
#             if not pdf_bytes:
#                 raise PDFParsingError(f"'{source_name}' is empty.")
#             doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     except PDFParsingError:
#         raise
#     except Exception as exc:  # PyMuPDF raises its own exception types
#         raise PDFParsingError(f"Could not open '{source_name}': it may be corrupted.") from exc

#     if doc.page_count == 0:
#         doc.close()
#         raise PDFParsingError(f"'{source_name}' has no pages.")

#     pages: list[PageText] = []
#     for page_index in range(doc.page_count):
#         page = doc.load_page(page_index)
#         raw_text = page.get_text("text")
#         if raw_text and raw_text.strip():
#             pages.append(PageText(page_number=page_index + 1, text=raw_text))
#     doc.close()

#     if not pages:
#         raise PDFParsingError(
#             f"No extractable text found in '{source_name}'. "
#             "It may be a scanned image without OCR text."
#         )

#     return pages


# def clean_text(raw_text: str) -> str:
#     """
#     Normalise whitespace, strip common PDF extraction artefacts (page
#     headers/footers repetition, hyphenated line breaks, control characters)
#     and collapse the text into readable paragraphs.
#     """
#     text = raw_text.replace("\x00", " ")

#     # Re-join words that were split across a line break with a hyphen,
#     # e.g. "informa-\ntion" -> "information".
#     text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

#     # Collapse remaining single newlines (soft wraps) into spaces, but keep
#     # paragraph breaks (double newlines) intact.
#     text = re.sub(r"\n{2,}", "\n\n", text)
#     text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

#     # Collapse repeated whitespace/tabs.
#     text = re.sub(r"[ \t]{2,}", " ", text)

#     # Strip stray non-printable characters.
#     text = re.sub(r"[^\x09\x0A\x20-\x7E\u00A0-\uFFFF]", "", text)

#     return text.strip()


# # A light heuristic set of leaflet section headers, used only to *tag* chunks
# # with a best-guess section label for source citation - it never filters or
# # blocks content.
# _SECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
#     ("Uses", re.compile(r"\bwhat .*(used for|is it for)\b", re.I)),
#     ("Dosage", re.compile(r"\b(how to take|dosage|dose)\b", re.I)),
#     ("Warnings", re.compile(r"\b(warning|precaution)s?\b", re.I)),
#     ("Side Effects", re.compile(r"\bside effects?\b", re.I)),
#     ("Storage", re.compile(r"\bstorage|how to store\b", re.I)),
#     ("Pregnancy", re.compile(r"\bpregnan(cy|t)\b", re.I)),
#     ("Overdose", re.compile(r"\boverdose\b", re.I)),
#     ("Missed Dose", re.compile(r"\bmissed dose\b", re.I)),
#     ("Ingredients", re.compile(r"\bingredients?|contains\b", re.I)),
# ]


# def _guess_section(text: str) -> str:
#     """Best-effort heuristic label for which leaflet section a chunk belongs to."""
#     for label, pattern in _SECTION_PATTERNS:
#         if pattern.search(text):
#             return label
#     return "General"


# def chunk_pages(pages: list[PageText], source_name: str) -> list[LeafletChunk]:
#     """
#     Split cleaned page text into overlapping chunks suitable for embedding,
#     preserving the originating page number on every chunk so answers can
#     always be traced back to a source page.
#     """
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=settings.chunk_size,
#         chunk_overlap=settings.chunk_overlap,
#         separators=["\n\n", "\n", ". ", " ", ""],
#     )

#     chunks: list[LeafletChunk] = []
#     running_index = 0
#     for page in pages:
#         cleaned = clean_text(page.text)
#         if not cleaned:
#             continue
#         for piece in splitter.split_text(cleaned):
#             piece = piece.strip()
#             if len(piece) < 10:
#                 continue  # skip near-empty fragments
#             running_index += 1
#             chunks.append(
#                 LeafletChunk(
#                     chunk_id=f"{source_name}-p{page.page_number}-c{running_index}",
#                     text=piece,
#                     page_number=page.page_number,
#                     source_file=source_name,
#                     section=_guess_section(piece),
#                 )
#             )

#     if not chunks:
#         raise PDFParsingError(f"'{source_name}' produced no usable text chunks after cleaning.")

#     return chunks


# def parse_pdf(pdf_source: Union[str, Path, BinaryIO], source_name: str) -> list[LeafletChunk]:
#     """Convenience wrapper: extract -> clean -> chunk in one call."""
#     pages = extract_pages(pdf_source, source_name)
#     return chunk_pages(pages, source_name)


# def full_text(pages: list[PageText]) -> str:
#     """Concatenate all pages into a single cleaned string (used for the
#     automatic leaflet summary, which needs broad context rather than a
#     narrow retrieved slice)."""
#     return "\n\n".join(clean_text(p.text) for p in pages)
"""
parser.py (FIXED - Arabic + English Support)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.config import settings


class PDFParsingError(Exception):
    pass


@dataclass
class PageText:
    page_number: int
    text: str


@dataclass
class LeafletChunk:
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    section: str = "General"


# ---------------------------
# PDF Extraction
# ---------------------------
def extract_pages(pdf_source, source_name) -> list[PageText]:
    try:
        if isinstance(pdf_source, (str, Path)):
            doc = fitz.open(pdf_source)
        else:
            pdf_bytes = pdf_source.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        raise PDFParsingError("Invalid PDF")

    pages = []
    for i in range(doc.page_count):
        text = doc.load_page(i).get_text("text")
        if text.strip():
            pages.append(PageText(i + 1, text))

    if not pages:
        raise PDFParsingError("No text found")

    return pages


# ---------------------------
# Cleaning (FIXED)
# ---------------------------
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")

    # لا تمسح كل ال new lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # إزالة رموز غريبة
    text = re.sub(r"[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\w\s.,:%()-]", " ", text)

    # مسافات
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------
# Section Detection (AR + EN)
# ---------------------------
_SECTION_PATTERNS = [
    ("Uses", re.compile(r"(used for|استعمال|يستخدم)", re.I)),
    ("Dosage", re.compile(r"(dosage|dose|جرعة)", re.I)),
    ("Warnings", re.compile(r"(warning|تحذير)", re.I)),
    ("Side Effects", re.compile(r"(side effects|آثار جانبية)", re.I)),
    ("Storage", re.compile(r"(storage|تخزين)", re.I)),
    ("Pregnancy", re.compile(r"(pregnancy|حمل)", re.I)),
]


def _guess_section(text: str) -> str:
    for name, pattern in _SECTION_PATTERNS:
        if pattern.search(text):
            return name
    return "General"


# ---------------------------
# Chunking (FIXED 🔥)
# ---------------------------
def chunk_pages(pages: list[PageText], source_name: str) -> list[LeafletChunk]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,              # 👈 أصغر = أدق
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            "؟",
            "!",
            ".",
            "،",   # عربي
            " ",
        ],
    )

    chunks = []
    idx = 0

    for page in pages:
        cleaned = clean_text(page.text)

        for piece in splitter.split_text(cleaned):
            if len(piece) < 30:
                continue

            idx += 1
            chunks.append(
                LeafletChunk(
                    chunk_id=f"{source_name}_{idx}",
                    text=piece,
                    page_number=page.page_number,
                    source_file=source_name,
                    section=_guess_section(piece),
                )
            )

    return chunks


def parse_pdf(pdf_source, source_name):
    pages = extract_pages(pdf_source, source_name)
    return chunk_pages(pages, source_name)


def full_text(pages: list[PageText]) -> str:
    return "\n\n".join(clean_text(p.text) for p in pages)