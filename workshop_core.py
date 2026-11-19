"""Shared source and result types for PDF search."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PDF_DATA_PATH = ROOT / "data"
QUERY_DATA_PATH = ROOT / "data" / "test_queries.json"


@dataclass(frozen=True)
class SourceFile:
    name: str
    data: bytes


PdfSource = SourceFile


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    body: str
    source_name: str
    page_number: int  # 1-based PDF page.
    search_text: str

    @property
    def location_label(self) -> str:
        return f"{self.source_name}, page {self.page_number}"

    @property
    def source_reference(self) -> str:
        return f"{self.source_name}:p{self.page_number}"


@dataclass(frozen=True)
class SearchHit:
    document: Document
    score: float
    method: str
