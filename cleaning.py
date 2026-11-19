"""Part 1 solution: extract and clean PDF pages."""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Sequence

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from workshop_core import Document, PDF_DATA_PATH, SourceFile


@dataclass(frozen=True)
class RawPdfPage:
    source_name: str
    source_digest: str
    page_number: int
    raw_text: str


def load_pdf_sources(directory: Path = PDF_DATA_PATH) -> list[SourceFile]:
    """Load every PDF under data/, including PDFs in nested folders."""
    paths = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".pdf"),
        key=lambda path: path.relative_to(directory).as_posix().casefold(),
    )
    basename_counts = Counter(path.name.casefold() for path in paths)
    return [
        SourceFile(
            path.relative_to(directory).as_posix() if basename_counts[path.name.casefold()] > 1 else path.name,
            path.read_bytes(),
        )
        for path in paths
    ]


def extract_pdf_pages(source: SourceFile) -> list[RawPdfPage]:
    """Read all pages, including pages that have no extractable text."""
    if not source.data.startswith(b"%PDF-"):
        raise ValueError(f"{source.name} is not a PDF file")
    try:
        reader = PdfReader(BytesIO(source.data))
        if reader.is_encrypted:
            raise ValueError(f"{source.name} is encrypted; use an unlocked PDF")
        digest = hashlib.sha256(source.data).hexdigest()[:8]
        return [
            RawPdfPage(source.name, digest, number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    except PdfReadError as error:
        raise ValueError(f"Could not read {source.name}: {error}") from error


def clean_pdf_text(raw_text: str) -> str:
    """Undo simple line-wrap noise and collapse whitespace."""
    text = raw_text.replace("\u00ad", "").replace("\r\n", "\n")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_page(page: RawPdfPage) -> Document | None:
    """Keep source and page metadata so every result can be checked."""
    body = clean_pdf_text(page.raw_text)
    if not body:
        return None
    first_line = next((line for line in page.raw_text.splitlines() if line.strip()), body)
    title = clean_pdf_text(first_line)[:120]
    source_path = Path(page.source_name).with_suffix("").as_posix()
    stem = re.sub(r"[^a-z0-9]+", "-", source_path.casefold()).strip("-") or "pdf"
    document_id = f"{stem}-{page.source_digest}-p{page.page_number}"
    return Document(document_id, title, body, page.source_name, page.page_number, body)


def _check_unique_names(sources: Sequence[SourceFile]) -> None:
    names = [source.name.casefold() for source in sources]
    if len(names) != len(set(names)):
        raise ValueError("Filenames must be unique; rename repeated files before uploading")


def audit_pdf_sources(sources: Sequence[SourceFile]) -> dict[str, int]:
    """Count pages that can and cannot enter the search index."""
    _check_unique_names(sources)
    if any(Path(source.name).suffix.casefold() != ".pdf" for source in sources):
        raise ValueError("Only PDF files are supported")
    pages = [page for source in sources for page in extract_pdf_pages(source)]
    searchable = sum(clean_page(page) is not None for page in pages)
    return {
        "files": len(sources),
        "pages": len(pages),
        "searchable_pages": searchable,
        "blank_pages": len(pages) - searchable,
    }


def load_clean_documents(sources: Sequence[SourceFile] | None = None) -> list[Document]:
    """Build one searchable document per text-bearing PDF page."""
    if sources is None:
        sources = load_pdf_sources()
    _check_unique_names(sources)
    if any(Path(source.name).suffix.casefold() != ".pdf" for source in sources):
        raise ValueError("Only PDF files are supported")
    documents = [
        document
        for source in sources
        for page in extract_pdf_pages(source)
        if (document := clean_page(page)) is not None
    ]
    if not documents:
        raise ValueError("No searchable text was found in the PDFs; scanned pages need OCR")
    return documents


if __name__ == "__main__":
    sample_sources = load_pdf_sources()
    print(audit_pdf_sources(sample_sources))
    documents = load_clean_documents(sample_sources)
    print(f"Searchable pages: {len(load_clean_documents(sample_sources))}")
    print( "\33[12;1m" + "hello" + "\33[m")
