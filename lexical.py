"""Part 2 solution: TF-IDF keyword search."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from workshop_core import Document, SearchHit
from cleaning import load_clean_documents


@dataclass
class LexicalIndex:
    documents: tuple[Document, ...]
    vectorizer: TfidfVectorizer
    matrix: spmatrix


def build_lexical_index(documents: Sequence[Document]) -> LexicalIndex:
    if not documents:
        raise ValueError("Cannot build an index from an empty corpus")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
    )
    texts = [document.search_text for document in documents]
    matrix = vectorizer.fit_transform(texts)
    return LexicalIndex(tuple(documents), vectorizer, matrix)


def search_lexical(index: LexicalIndex, query: str, k: int = 5) -> list[SearchHit]:
    if not query.strip() or k <= 0:
        return []
    query_vector = index.vectorizer.transform([query])
    # TfidfVectorizer L2-normalises vectors by default, so dot product is cosine similarity.
    scores = (index.matrix @ query_vector.T).toarray().ravel()
    ranked = np.argsort(-scores, kind="stable")
    return [
        SearchHit(index.documents[i], float(scores[i]), "lexical")
        for i in ranked[:k]
        if scores[i] > 0
    ]


if __name__ == "__main__":
    index = build_lexical_index(load_clean_documents())
    
    search = input("Enter the search text")
    
    print("Searching for \"" + search + "\"")
    
    for hit in search_lexical(index, "ASLR bypass"):
        print(f"{hit.score:.3f} {hit.document.document_id} {hit.document.title}")
        print(f"\33[31;1m Matched text: \33[0m \n" + hit.document.body)
