# TASK-1:

# Automated Webinar Transcript Summarization & QA System

## Executive Summary

Emily, a marketing manager at a tech startup, recently attended a comprehensive multi-part webinar series covering key topics including that is already given in the Project.Problem.Requirement.files:

- **Content Marketing Strategies**
- **Data Analytics for Marketers**
- **Social Media Advertising**
- **Advanced Strategies for Social Media Marketing**

Because each session ran for 3 to 4 hours, the resulting transcript files are exceptionally lengthy and dense. To enable Emily to efficiently extract key insights and share them across her team, this project aims to build an automated **Document Summarization and Interactive Question-Answering (QA) System** leveraging **LangChain**, **FAISS**, and **Ollama**.

---

## Technical Architecture & Stack

| Component | Technology | Description |
| --- | --- | --- |
| **LLM & Agent Framework** | **LangChain** | Orchestrates document loaders, prompt templates, and retrieval chains. |
| **Local LLM** | **Ollama** | Serves the any model locally via `ChatOllama` for privacy and cost-efficiency. |
| **Embeddings** | **HuggingFace / Ollama** | Generates vector representations of document chunks (`all-MiniLM-L6-v2` or `nomic-embed-text`). |
| **Vector Store** | **FAISS** | Facebook AI Similarity Search database for fast similarity retrieval and metadata management. |
| **User Interface** | **Streamlit** | Python-native Web UI for document upload, processing status, summary display, and interactive chat. |

---

## Key Functional Requirements

### 1. Document Loading, Processing & Vector Storage

- **Document Upload:** Support uploading webinar transcript PDFs where the filename or title reflects the webinar name.
- **Scraping & Chunking:** Extract text from the PDF and split it into manageable chunks using text splitters (e.g., `RecursiveCharacterTextSplitter`).
- **Vector Indexing:** Index transcript chunks into a FAISS vector store.
- **Automated Summarization:**
    - Utilize a structured `PromptTemplate` and LLM chain to generate a concise, high-level summary of the entire webinar transcript.
    - Store the generated summary into FAISS along with metadata (e.g., webinar title, timestamp, document ID) for future reference and retrieval.

### 2. Dashboard & Visualizations

- **File Upload Area:** A simple drag-and-drop widget to upload transcript PDFs and initiate ingestion.
- **Summary Display:** Clean UI layout displaying the generated concise summary immediately upon processing.
- **FAISS Vector Store Inspector:** Display indexed metadata and stored summaries currently housed inside the FAISS database.

### 3. Interactive Question Answering (QA)

- **Query Input:** An interactive chat bar allowing users to ask specific questions regarding the transcript content.
- **Contextual QA Chain:** Route user prompts through a LangChain retrieval pipeline (`RetrievalQA` or LCEL-based chain) backed by `ChatOllama` running the `llama-3` model.
- **Response Rendering:** Display precise answers contextualized against the uploaded document.

---

## Project Structure (Example-Layout)

```
webinar-summarizer/
├── README.md               # Project documentation and intern instructions
├── requirements.txt        # Python library dependencies
├── app.py                  # Main Streamlit UI entry point
├── config.py               # Application configuration and model settings
└── src/
    ├── __init__.py
    ├── loader.py           # Document loading and text chunking logic
    ├── vector_store.py     # FAISS database management & metadata indexing
    ├── summarizer.py       # Prompt templates & summary chain logic
    └── qa_engine.py        # ChatOllama retrieval & response generation logic
```

---

### Task 1: Document Processing & FAISS Storage (`loader.py` & `vector_store.py`)

- Implement `PyPDFLoader` to parse transcript PDFs.
- Split documents into chunks (~1000 characters with 200 overlap).
- Initialize FAISS store with local embeddings (e.g., `HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")`).

### Task 2: Summarization Engine (`summarizer.py`)

- Design a concise `PromptTemplate` focusing on marketing takeaways, key frameworks, and actionable insights.
- Build a map-reduce or refine chain for long documents using LangChain primitives.
- Persist the generated summary back into FAISS attached with the webinar title metadata.

### Task 3: Contextual QA Engine (`qa_engine.py`)

- Instantiate `ChatOllama(model="llama3", temperature=0.2)`.
- Create a retrieval chain combining similarity search from FAISS with a question-answering prompt.

### Task 4: Streamlit Interface (`app.py`) - Optional(Good to have)

- Build a multi-panel UI:
    - **Sidebar:** Document upload + processing status button.
    - **Main Area Top:** Generated Summary view & FAISS store record inspector.
    - **Main Area Bottom:** Chat component for interactive QA (`st.chat_input` / `st.chat_message`).

---

---

## Definition of Done (Submission Criteria)

- [ ]  Code is organized cleanly under `src/` modular structure.
- [ ]  Application successfully loads a 50+ page transcript PDF without memory issues.
- [ ]  Concise summary is generated, saved into FAISS with metadata, and displayed on the UI.
- [ ]  Chat query interface successfully answers domain questions based on transcript context using any appropriate model of choice.
