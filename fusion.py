"""Part 5 solution: reciprocal rank fusion for hybrid search."""

from collections.abc import Sequence

from workshop_core import Document, SearchHit
from cleaning import load_clean_documents
from lexical import LexicalIndex, build_lexical_index, search_lexical
from semantic import SemanticIndex, build_semantic_index, search_semantic


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchHit]], limit: int = 5, rank_constant: int = 60
) -> list[SearchHit]:
    if limit <= 0:
        return []
    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for ranking in rankings:
        seen_in_this_ranking: set[str] = set()
        for rank, hit in enumerate(ranking, start=1):
            document_id = hit.document.document_id
            if document_id in seen_in_this_ranking:
                continue
            seen_in_this_ranking.add(document_id)
            documents[document_id] = hit.document
            scores[document_id] = scores.get(document_id, 0.0) + 1 / (
                rank_constant + rank
            )
    ranked_ids = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return [
        SearchHit(documents[document_id], scores[document_id], "hybrid")
        for document_id in ranked_ids[:limit]
    ]


def search_hybrid(
    lexical_index: LexicalIndex,
    semantic_index: SemanticIndex,
    query: str,
    k: int = 5,
    candidates: int = 20,
) -> list[SearchHit]:
    lexical_ids = [doc.document_id for doc in lexical_index.documents]
    semantic_ids = [doc.document_id for doc in semantic_index.documents]
    if lexical_ids != semantic_ids:
        raise ValueError("The keyword and semantic indexes must use the same documents")
    if not query.strip() or k <= 0:
        return []
    candidate_count = max(k, candidates)
    keyword_hits = search_lexical(lexical_index, query, candidate_count)
    semantic_hits = search_semantic(semantic_index, query, candidate_count)
    return reciprocal_rank_fusion([keyword_hits, semantic_hits], limit=k)


if __name__ == "__main__":
    documents = load_clean_documents()
    lexical_index = build_lexical_index(documents)
    semantic_index = build_semantic_index(documents)
    for hit in search_hybrid(lexical_index, semantic_index, "HTTP 429"):
        print(f"{hit.score:.4f} {hit.document.document_id} {hit.document.title}")
