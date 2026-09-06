"""
loader.py
---------
Handles loading webinar transcript PDFs and splitting them into
retrieval-friendly chunks.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader

try:
    # Newer LangChain versions ship the splitter in its own package.
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    # Older LangChain versions expose it under langchain.text_splitter.
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

import config

#This creates a custom data structure for your webinar.
@dataclass
class WebinarDocument:
    """Container describing an ingested webinar and its chunks."""

    doc_id: str
    title: str
    source_path: str
    uploaded_at: str
    page_count: int
    chunks: List[Document] = field(default_factory=list)


def _derive_title(filename: str) -> str:
    """Turn an uploaded filename into a human-readable webinar title."""
    stem = Path(filename).stem
    # Strip any leading numeric/timestamp prefix like "1788372891353_"
    parts = stem.split("_", 1)
    if len(parts) == 2 and parts[0].isdigit():
        stem = parts[1]
    return stem.replace("_", " ").replace("-", " ").strip().title()


def _make_doc_id(filename: str, content_hint: str = "") -> str:
    digest_source = f"{filename}-{content_hint}".encode("utf-8", errors="ignore")
    return hashlib.sha1(digest_source).hexdigest()[:12]


def save_uploaded_file(uploaded_file) -> Path:
    """Persist a Streamlit UploadedFile object to disk and return its path."""
    dest = config.UPLOAD_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest


def load_and_chunk_pdf(pdf_path: str | Path, title: str | None = None) -> WebinarDocument:
    """
    Load a transcript PDF with PyPDFLoader and split it into overlapping
    chunks using RecursiveCharacterTextSplitter.

    Parameters
    ----------
    pdf_path : path to the PDF file on disk
    title    : optional human-readable webinar title. If omitted, it is
               derived from the filename.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    webinar_title = title or _derive_title(pdf_path.name)
    doc_id = _make_doc_id(pdf_path.name)

    loader = PyPDFLoader(str(pdf_path))
    raw_pages = loader.load()  # one Document per page, keeps memory low

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_pages)

    # Attach rich metadata to every chunk so it can be traced back later.
    for i, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "doc_id": doc_id,
                "webinar_title": webinar_title,
                "source_file": pdf_path.name,
                "chunk_index": i,
            }
        )

    return WebinarDocument(
        doc_id=doc_id,
        title=webinar_title,
        source_path=str(pdf_path),
        uploaded_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        page_count=len(raw_pages),
        chunks=chunks,
    )
