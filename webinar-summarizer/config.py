"""
config.py
---------
Central configuration for the Webinar Summarizer & QA System.
Edit the values below (or set matching environment variables) to change
models, chunk sizes, or storage locations without touching app logic.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
SUMMARY_STORE_PATH = DATA_DIR / "summaries.json"

for _dir in (DATA_DIR, UPLOAD_DIR, FAISS_INDEX_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Ollama / LLM settings
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# "huggingface" -> local sentence-transformers model (no Ollama pull needed)
# "ollama"      -> uses an Ollama embedding model (e.g. nomic-embed-text)
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
OLLAMA_EMBEDDING_MODEL = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
# "stuff" (single call, fastest for short/medium transcripts), "map_reduce"
# (scales to very long transcripts), "refine" (narrative, chunk by chunk),
# or "auto" (stuff when the transcript fits in context, else map_reduce).
SUMMARY_CHAIN_TYPE = os.getenv("SUMMARY_CHAIN_TYPE", "auto")

# ---------------------------------------------------------------------------
# Retrieval QA
# ---------------------------------------------------------------------------
RETRIEVER_TOP_K = int(os.getenv("RETRIEVER_TOP_K", "4"))
