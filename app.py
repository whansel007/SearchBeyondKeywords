"""Standalone app: upload PDFs and inspect page-level search results.

Run from the project root:
    python -m streamlit run combined/app.py
"""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Sequence

import streamlit as st

from workshop_core import SourceFile, SearchHit
from cleaning import audit_pdf_sources, load_clean_documents, load_pdf_sources
from lexical import LexicalIndex, build_lexical_index, search_lexical
from semantic import SemanticIndex, build_semantic_index, load_encoder, search_semantic
from fusion import search_hybrid


@st.cache_resource(show_spinner="Loading the embedding model...")
def cached_encoder():
    return load_encoder()


def load_indexes(sources: Sequence[SourceFile]):
    """Keep uploaded document indexes within this browser session."""
    key = tuple((source.name, hashlib.sha256(source.data).hexdigest()) for source in sources)
    if st.session_state.get("source_index_key") != key:
        audit = audit_pdf_sources(sources)
        documents = load_clean_documents(sources)
        lexical_index = build_lexical_index(documents)
        semantic_index = build_semantic_index(documents, encoder=cached_encoder())
        st.session_state["source_index_key"] = key
        st.session_state["source_indexes"] = (audit, documents, lexical_index, semantic_index)
    return st.session_state["source_indexes"]


def run_search(lexical_index, semantic_index, query: str, method: str) -> list[SearchHit]:
    if method == "Keyword":
        return search_lexical(lexical_index, query, k=5)
    if method == "Semantic":
        return search_semantic(semantic_index, query, k=5)
    if method == "Hybrid":
        return search_hybrid(lexical_index, semantic_index, query, k=5)
    raise ValueError(f"Unknown ranking method: {method}")


def show_hit(hit: SearchHit, position: int) -> None:
    document = hit.document
    st.subheader(f"{position}. {document.title}")
    st.caption(
        f"{document.location_label} · "
        f"{hit.method} score {hit.score:.3f}"
    )
    st.write(document.body)


def main(
    *,
    title: str = "Search your PDFs",
    description: str = "Upload text-based PDFs, then compare keyword, semantic, and hybrid results by page.",
    after_search: Callable[[str, LexicalIndex, SemanticIndex], None] | None = None,
) -> None:
    st.title(title)
    st.write(description)
    uploaded_files = st.file_uploader(
        "Upload PDF files", type="pdf", accept_multiple_files=True,
        help="Uploaded files replace the PDFs under data/ for this session.",
    )
    sources = (
        tuple(SourceFile(Path(file.name).name, file.getvalue()) for file in uploaded_files)
        if uploaded_files else tuple(load_pdf_sources())
    )
    if not uploaded_files:
        st.caption(f"Using {len(sources)} PDFs from data/. Upload PDFs to search your own files.")
    try:
        with st.spinner("Indexing PDF pages..."):
            audit, documents, lexical_index, semantic_index = load_indexes(sources)
    except ValueError as error:
        st.error(str(error))
        return
    st.caption(f"Indexed {len(documents)} text pages from {len(sources)} PDFs.")
    if audit["blank_pages"]:
        st.warning(f"Skipped {audit['blank_pages']} page(s) with no extractable text. Scanned pages need OCR.")
    st.caption("Scores use different scales; compare result order and read the source text.")

    with st.form("pdf_search", enter_to_submit=True):
        query = st.text_input("Search", value="How does RAG improve LLMs?")
        method = st.selectbox("Ranking method", ["Hybrid", "Keyword", "Semantic"])
        submitted = st.form_submit_button("Search")

    if submitted:
        st.session_state["submitted_search"] = (query.strip(), method)
    if "submitted_search" not in st.session_state:
        st.info("Enter a question and press Search.")
        return

    query, method = st.session_state["submitted_search"]
    if not query:
        st.info("Enter a question to search.")
        return
    st.caption(f"Results for: {query} · {method}")
    with st.spinner("Searching PDF pages..."):
        hits = run_search(lexical_index, semantic_index, query, method)

    if after_search is not None:
        after_search(query, lexical_index, semantic_index)

    if hits:
        for position, hit in enumerate(hits, start=1):
            show_hit(hit, position)
    else:
        st.info("No matching pages were found for this ranking method.")

    if not hits:
        return

    source_by_name = {source.name: source for source in sources}
    choices = list(dict.fromkeys(hit.document.source_name for hit in hits))
    selected = st.selectbox("Preview source PDF", choices)
    relevant_pages = [
        str(hit.document.page_number) for hit in hits if hit.document.source_name == selected
    ]
    st.caption(f"Matching page(s) in {selected}: {', '.join(relevant_pages)}. Use the viewer's page control.")
    st.pdf(source_by_name[selected].data, height=500)
    st.download_button(
        "Download this PDF", source_by_name[selected].data,
        file_name=selected,
        mime="application/pdf",
    )


if __name__ == "__main__":
    main()
