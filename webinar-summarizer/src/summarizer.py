"""
summarizer.py
-------------
Prompt templates and a map-reduce / refine summarization pipeline used to
turn a long webinar transcript into a concise, marketing-focused summary.

Implemented with plain LCEL (prompt | llm | parser) instead of the legacy
`langchain.chains.summarize` module, since that module's location and
internals have shifted across LangChain releases and depending on it makes
the app fragile to version drift. This version only depends on the stable
`langchain_core` primitives plus `langchain_ollama`.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:  # pragma: no cover - older langchain fallback
    from langchain.prompts import PromptTemplate

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - older langchain fallback
    from langchain.schema import Document

import config

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MAP_PROMPT = PromptTemplate(
    template=(
        "You are a marketing analyst. Read the following excerpt from a "
        "webinar transcript and extract the key marketing takeaways, "
        "frameworks, and any actionable tactics mentioned. Be concise and "
        "use bullet points.\n\n"
        "Transcript excerpt:\n{text}\n\n"
        "Key points:"
    ),
    input_variables=["text"],
)

COMBINE_PROMPT = PromptTemplate(
    template=(
        "You are a senior marketing strategist preparing a briefing for a "
        "busy marketing manager. Combine the bullet points below - drawn "
        "from different parts of the same webinar - into ONE concise, "
        "well-organized summary with this structure:\n\n"
        "1. Overview (2-3 sentences)\n"
        "2. Key Frameworks & Concepts (bullet points)\n"
        "3. Actionable Takeaways (bullet points)\n\n"
        "Avoid repetition and keep it tight and skimmable.\n\n"
        "Notes:\n{text}\n\n"
        "Final Summary:"
    ),
    input_variables=["text"],
)

REFINE_INITIAL_PROMPT = PromptTemplate(
    template=(
        "You are a marketing analyst. Write a concise summary of the "
        "following webinar transcript excerpt, focusing on marketing "
        "takeaways, frameworks, and actionable insights.\n\n{text}\n\n"
        "SUMMARY:"
    ),
    input_variables=["text"],
)

REFINE_PROMPT = PromptTemplate(
    template=(
        "You are refining a running summary of a marketing webinar as more "
        "transcript context becomes available.\n"
        "Existing summary:\n{existing_answer}\n\n"
        "New transcript context:\n{text}\n\n"
        "Refine the existing summary (only if the new context adds useful "
        "marketing takeaways, frameworks, or actionable insights). Keep it "
        "concise and organized under: Overview, Key Frameworks & Concepts, "
        "Actionable Takeaways.\n\n"
        "Refined summary:"
    ),
    input_variables=["existing_answer", "text"],
)

STUFF_PROMPT = PromptTemplate(
    template=(
        "You are a senior marketing strategist. Read the entire webinar "
        "transcript below and produce ONE concise, well-organized summary "
        "with this structure:\n\n"
        "1. Overview (2-3 sentences)\n"
        "2. Key Frameworks & Concepts (bullet points)\n"
        "3. Actionable Takeaways (bullet points)\n\n"
        "Keep it tight and skimmable - avoid repeating the transcript "
        "verbatim.\n\n"
        "Transcript:\n{text}\n\n"
        "Summary:"
    ),
    input_variables=["text"],
)

# Roughly how many characters worth of "key point" notes to combine per
# reduce pass, so the combine step itself never overflows context on very
# long transcripts (50+ pages -> potentially 100+ map outputs).
MAX_CHARS_PER_COMBINE_BATCH = 6000

# If the whole transcript fits comfortably in one prompt (small decks, short
# transcripts, the ~9-page sample dataset, etc.), a single "stuff" call is
# both faster and higher quality than map-reduce, which only pays off once
# a document is too long to fit in context in one shot (50+ pages).
STUFF_CHAR_THRESHOLD = 12000


def _get_llm(num_predict: int = 400) -> ChatOllama:
    return ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        temperature=config.OLLAMA_TEMPERATURE,
        base_url=config.OLLAMA_BASE_URL,
        timeout=120,  # fail fast instead of hanging forever if Ollama is unreachable/stuck
        num_predict=num_predict,  # cap output tokens so calls don't ramble on and slow things down
    )


def _batch_by_char_budget(texts: List[str], budget: int) -> List[List[str]]:
    """Group texts into batches that stay under a rough character budget."""
    batches: List[List[str]] = []
    current: List[str] = []
    current_len = 0
    for t in texts:
        if current and current_len + len(t) > budget:
            batches.append(current)
            current = []
            current_len = 0
        current.append(t)
        current_len += len(t)
    if current:
        batches.append(current)
    return batches


def _map_reduce_summarize(chunks: List[Document], llm: ChatOllama) -> str:
    parser = StrOutputParser()
    map_chain = MAP_PROMPT | llm | parser
    combine_chain = COMBINE_PROMPT | llm | parser

    # Map: summarize each chunk independently. These calls don't depend on
    # each other, so run them concurrently - Ollama will queue/parallelize
    # what it can, which is faster than waiting on each one sequentially.
    with ThreadPoolExecutor(max_workers=4) as pool:
        partial_notes = list(pool.map(lambda c: map_chain.invoke({"text": c.page_content}), chunks))

    # Reduce: repeatedly combine batches of notes until only one remains,
    # so this scales to very long transcripts without overflowing context.
    while len(partial_notes) > 1:
        batches = _batch_by_char_budget(partial_notes, MAX_CHARS_PER_COMBINE_BATCH)
        partial_notes = [
            combine_chain.invoke({"text": "\n\n".join(batch)}) for batch in batches
        ]

    return partial_notes[0].strip()


def _refine_summarize(chunks: List[Document], llm: ChatOllama) -> str:
    parser = StrOutputParser()
    initial_chain = REFINE_INITIAL_PROMPT | llm | parser
    refine_chain = REFINE_PROMPT | llm | parser

    if not chunks:
        return ""

    summary = initial_chain.invoke({"text": chunks[0].page_content})
    for chunk in chunks[1:]:
        summary = refine_chain.invoke(
            {"existing_answer": summary, "text": chunk.page_content}
        )
    return summary.strip()


def _stuff_summarize(chunks: List[Document], llm: ChatOllama) -> str:
    """Single-call summarization: pass the whole transcript in one prompt.
    Fastest and highest-quality option when the document is short enough
    to fit in context (e.g. the ~9-page sample dataset)."""
    parser = StrOutputParser()
    chain = STUFF_PROMPT | llm | parser
    full_text = "\n\n".join(c.page_content for c in chunks)
    return chain.invoke({"text": full_text}).strip()


def summarize_chunks(chunks: List[Document], chain_type: str | None = None) -> str:
    """
    Summarize a list of transcript chunks using "stuff" (single call, fast
    - best for short/medium transcripts), "map_reduce" (scales to very long
    transcripts), or "refine" (narrative pass, chunk by chunk).

    If chain_type is left as "auto" (the default), the whole transcript is
    stuffed into one call when it's short enough to fit comfortably in
    context, and falls back to map_reduce for longer documents.
    """
    if not chunks:
        return "No content available to summarize."

    chain_type = chain_type or config.SUMMARY_CHAIN_TYPE
    llm = _get_llm()

    total_chars = sum(len(c.page_content) for c in chunks)

    if chain_type == "auto":
        chain_type = "stuff" if total_chars <= STUFF_CHAR_THRESHOLD else "map_reduce"

    if chain_type == "stuff":
        return _stuff_summarize(chunks, llm)
    if chain_type == "refine":
        return _refine_summarize(chunks, llm)
    return _map_reduce_summarize(chunks, llm)
