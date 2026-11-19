"""Part 4 solution: sentence-embedding semantic search."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from workshop_core import Document, SearchHit
from cleaning import load_clean_documents


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SemanticIndex:
    documents: tuple[Document, ...]
    encoder: Any
    vectors: np.ndarray


def load_encoder(model_name: str = MODEL_NAME) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(lengths, 1e-12)


def build_semantic_index(
    documents: Sequence[Document], encoder: Any | None = None
) -> SemanticIndex:
    if not documents:
        raise ValueError("Cannot build an index from an empty corpus")
    if encoder is None:
        encoder = load_encoder()
    texts = [document.search_text for document in documents]
    vectors = _unit_rows(encoder.encode_document(texts, convert_to_numpy=True))
    if len(vectors) != len(documents):
        raise ValueError("Encoder returned a different number of vectors than documents")
    return SemanticIndex(tuple(documents), encoder, vectors)


def search_semantic(index: SemanticIndex, query: str, k: int = 5) -> list[SearchHit]:
    if not query.strip() or k <= 0:
        return []
    query_vector = _unit_rows(index.encoder.encode_query(query, convert_to_numpy=True))[0]
    scores = index.vectors @ query_vector
    ranked = np.argsort(-scores, kind="stable")[:k]
    return [
        SearchHit(index.documents[i], float(scores[i]), "semantic") for i in ranked
    ]


if __name__ == "__main__":
    index = build_semantic_index(load_clean_documents())
    for hit in search_semantic(index, "how does transformers make LLM generate better replies?"):
        print(f"{hit.score:.3f} {hit.document.document_id} {hit.document.title}")
