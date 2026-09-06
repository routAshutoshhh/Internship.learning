"""
vector_store.py
----------------
Manages the FAISS vector store: index creation, persistence, chunk
ingestion, summary storage, and metadata inspection.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

import config

_lock = threading.Lock()
_embeddings = None


def get_embeddings():
    """Lazily build (and cache) the embeddings model configured in config.py."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if config.EMBEDDING_PROVIDER == "ollama":
        from langchain_ollama import OllamaEmbeddings

        _embeddings = OllamaEmbeddings(
            model=config.OLLAMA_EMBEDDING_MODEL,
            base_url=config.OLLAMA_BASE_URL,
        )
    else:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(model_name=config.HF_EMBEDDING_MODEL)

    return _embeddings


def _index_paths() -> tuple[Path, Path]:
    faiss_file = config.FAISS_INDEX_DIR / "index.faiss"
    pkl_file = config.FAISS_INDEX_DIR / "index.pkl"
    return faiss_file, pkl_file


def load_index() -> Optional[FAISS]:
    """Load the persisted FAISS index from disk, if it exists."""
    faiss_file, pkl_file = _index_paths()
    if not (faiss_file.exists() and pkl_file.exists()):
        return None
    return FAISS.load_local(
        str(config.FAISS_INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def save_index(store: FAISS) -> None:
    store.save_local(str(config.FAISS_INDEX_DIR))


def add_chunks(chunks: List[Document]) -> FAISS:
    """Add transcript chunks to the FAISS store, creating it if needed."""
    with _lock:
        store = load_index()
        if store is None:
            store = FAISS.from_documents(chunks, get_embeddings())
        else:
            store.add_documents(chunks)
        save_index(store)
        return store


def add_summary(doc_id: str, webinar_title: str, summary_text: str, source_file: str, uploaded_at: str) -> FAISS:
    """
    Persist the generated summary both as a searchable FAISS document
    (so the QA chain can answer "what was this webinar about") and as a
    lightweight JSON record for the dashboard inspector.
    """
    summary_doc = Document(
        page_content=summary_text,
        metadata={
            "doc_id": doc_id,
            "webinar_title": webinar_title,
            "source_file": source_file,
            "type": "summary",
        },
    )
    store = add_chunks([summary_doc])
    _append_summary_record(doc_id, webinar_title, summary_text, source_file, uploaded_at)
    return store


def _append_summary_record(doc_id: str, webinar_title: str, summary_text: str, source_file: str, uploaded_at: str) -> None:
    records = read_summary_records()
    records = [r for r in records if r["doc_id"] != doc_id]  # replace if re-ingested
    records.append(
        {
            "doc_id": doc_id,
            "webinar_title": webinar_title,
            "summary": summary_text,
            "source_file": source_file,
            "uploaded_at": uploaded_at,
        }
    )
    with open(config.SUMMARY_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def read_summary_records() -> List[dict]:
    if not config.SUMMARY_STORE_PATH.exists():
        return []
    with open(config.SUMMARY_STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def inspect_metadata(limit: int = 200) -> List[dict]:
    """Return a flat list of metadata dicts for chunks currently indexed."""
    store = load_index()
    if store is None:
        return []
    out = []
    docstore = store.docstore
    ids = list(store.index_to_docstore_id.values())[:limit]
    for doc_id in ids:
        doc = docstore.search(doc_id)
        if doc is not None:
            meta = dict(doc.metadata)
            meta["preview"] = doc.page_content[:160].replace("\n", " ")
            out.append(meta)
    return out


def as_retriever(k: int = None):
    store = load_index()
    if store is None:
        return None
    return store.as_retriever(search_kwargs={"k": k or config.RETRIEVER_TOP_K})


def total_chunk_count() -> int:
    """Return how many vectors/chunks are currently indexed in FAISS."""
    store = load_index()
    if store is None:
        return 0
    return len(store.index_to_docstore_id)


def reset_store() -> None:
    """Wipe the FAISS index and the summary records, for a clean restart."""
    import shutil

    with _lock:
        if config.FAISS_INDEX_DIR.exists():
            shutil.rmtree(config.FAISS_INDEX_DIR)
            config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
        if config.SUMMARY_STORE_PATH.exists():
            config.SUMMARY_STORE_PATH.unlink()
