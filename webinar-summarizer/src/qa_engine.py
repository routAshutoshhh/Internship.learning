"""
qa_engine.py
------------
Contextual Question-Answering engine that combines FAISS similarity
retrieval with a ChatOllama LLM (LCEL-style chain), so users can ask
free-form questions about the ingested webinar transcripts.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

import config
from src.vector_store import as_retriever

QA_SYSTEM_PROMPT = (
    "You are a helpful marketing research assistant. Answer the user's "
    "question using ONLY the provided webinar transcript context. If the "
    "context does not contain the answer, say you don't have enough "
    "information from the ingested transcripts rather than guessing. "
    "Where useful, mention which webinar the information came from.\n\n"
    "Context:\n{context}"
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    formatted = []
    for d in docs:
        title = d.metadata.get("webinar_title", "Unknown Webinar")
        formatted.append(f"[{title}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)


def _get_llm() -> ChatOllama:
    return ChatOllama(
        model=config.OLLAMA_CHAT_MODEL,
        temperature=config.OLLAMA_TEMPERATURE,
        base_url=config.OLLAMA_BASE_URL,
        timeout=120,
        num_predict=500,
    )


def answer_question(question: str, k: int = None) -> Tuple[str, List[Document]]:
    """
    Run a retrieval-augmented QA pass over the FAISS store.

    Returns
    -------
    (answer_text, source_documents)
    """
    retriever = as_retriever(k=k)
    if retriever is None:
        return (
            "No documents have been indexed yet. Please upload and process "
            "a webinar transcript first.",
            [],
        )

    source_docs = retriever.invoke(question)
    llm = _get_llm()

    chain = (
        {
            "context": lambda x: _format_docs(source_docs),
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    return answer.strip(), source_docs
