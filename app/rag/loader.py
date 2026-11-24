"""Extract searchable text while preserving original PDF page numbers."""

import re
import unicodedata
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.core.errors import AppError

SUPPORTED_TYPES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(frozen=True)
class Page:
    text: str
    number: int | None = None


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n(?: *\n)+", "\n\n", text)
    return text.strip()


def validate_filename(filename: str) -> str:
    name = PurePosixPath(filename.replace("\\", "/")).name
    if PurePosixPath(name).suffix.lower() not in SUPPORTED_TYPES:
        raise AppError(415, "Upload a PDF, TXT, or Markdown file.")
    if len(name) > 255 or any(ord(char) < 32 for char in name):
        raise AppError(400, "The filename is invalid.")
    return name


def load_document(filename: str, content: bytes) -> list[Page]:
    """Read one UTF-8 text file or the text layers of a PDF."""
    filename = validate_filename(filename)
    if filename.lower().endswith(".pdf"):
        if b"%PDF-" not in content[:1024]:
            raise AppError(422, "The file does not have a valid PDF header.")
        try:
            reader = PdfReader(BytesIO(content))
            if reader.is_encrypted:
                raise AppError(422, "Upload an unencrypted PDF.")
            pages = [
                Page(normalize_text(page.extract_text() or ""), number)
                for number, page in enumerate(reader.pages, start=1)
            ]
        except (PyPdfError, ValueError, KeyError, TypeError, RecursionError, OverflowError) as error:
            raise AppError(422, "The PDF could not be read.") from error
    else:
        try:
            decoded = content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise AppError(422, "Text and Markdown uploads must use UTF-8.") from error
        if any(ord(char) < 32 and char not in "\t\n\r\f" for char in decoded):
            raise AppError(422, "The upload contains binary data.")
        pages = [Page(normalize_text(decoded))]
    result = [page for page in pages if page.text]
    if not result:
        raise AppError(422, "No extractable text was found. Scanned PDFs need a text layer.")
    return result
