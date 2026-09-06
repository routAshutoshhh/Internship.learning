"""
app.py
------
Streamlit front-end for the Webinar Transcript Summarizer & QA System.

Run with:
    streamlit run app.py
"""

import traceback
from datetime import datetime

import streamlit as st

import config
from src.loader import save_uploaded_file, load_and_chunk_pdf
from src.vector_store import (
    add_chunks,
    add_summary,
    read_summary_records,
    inspect_metadata,
    total_chunk_count,
    reset_store,
)
from src.summarizer import summarize_chunks
from src.qa_engine import answer_question

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Webinar Insight Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom styling — deep teal / indigo palette, high-contrast text
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(160deg, #0B1220 0%, #0E1A2B 50%, #0A1420 100%);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0F1B2D 0%, #0A1420 100%);
            border-right: 1px solid rgba(45, 212, 191, 0.12);
        }

        .hero {
            padding: 2.1rem 2.4rem;
            border-radius: 22px;
            background: linear-gradient(120deg, #0EA5A5 0%, #2563EB 60%, #7C3AED 120%);
            box-shadow: 0 18px 40px -14px rgba(14, 165, 165, 0.45);
            margin-bottom: 1.6rem;
        }
        .hero h1 {
            font-family: 'Sora', sans-serif;
            color: white;
            font-size: 2.1rem;
            font-weight: 800;
            margin: 0 0 0.35rem 0;
            letter-spacing: -0.5px;
        }
        .hero p {
            color: rgba(255,255,255,0.95);
            font-size: 1.02rem;
            margin: 0;
            max-width: 720px;
        }

        .glass-card {
            background: rgba(45, 212, 191, 0.05);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 16px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.1rem;
        }

        .section-title {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: #E2E8F0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }

        .pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            margin-right: 0.4rem;
            margin-bottom: 0.3rem;
        }
        .pill-teal   { background: rgba(20,184,166,0.16); color: #5EEAD4; border: 1px solid rgba(20,184,166,0.4);}
        .pill-blue   { background: rgba(59,130,246,0.16); color: #93C5FD; border: 1px solid rgba(59,130,246,0.4);}
        .pill-amber  { background: rgba(245,158,11,0.16); color: #FCD34D; border: 1px solid rgba(245,158,11,0.4);}

        .summary-card {
            background: linear-gradient(160deg, rgba(14,165,165,0.14), rgba(37,99,235,0.08));
            border: 1px solid rgba(45,212,191,0.3);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
            white-space: pre-wrap;
            color: #F1F5F9;
            line-height: 1.6;
        }

        .meta-row {
            font-size: 0.8rem;
            color: #5EEAD4;
            margin-bottom: 0.6rem;
        }

        .stButton>button {
            background: linear-gradient(120deg, #0EA5A5, #2563EB);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            transition: transform 0.15s ease;
        }
        .stButton>button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px -10px rgba(37,99,235,0.6);
        }

        [data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        .chat-panel-title {
            font-family: 'Sora', sans-serif;
            font-weight: 700;
            font-size: 1.05rem;
            color: #F1F5F9;
            padding: 0.9rem 1.1rem;
            background: linear-gradient(120deg, rgba(14,165,165,0.25), rgba(37,99,235,0.18));
            border-radius: 14px 14px 0 0;
            border: 1px solid rgba(45,212,191,0.25);
            border-bottom: none;
        }

        .stat-row {
            display: flex;
            gap: 0.6rem;
            margin-bottom: 0.8rem;
        }
        .stat-box {
            flex: 1;
            background: rgba(45, 212, 191, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 0.7rem 0.6rem;
            text-align: center;
        }
        .stat-box .stat-value {
            font-family: 'Sora', sans-serif;
            font-size: 1.4rem;
            font-weight: 800;
            color: #5EEAD4;
            line-height: 1.1;
        }
        .stat-box .stat-label {
            font-size: 0.68rem;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-top: 0.2rem;
        }

        .footer-note {
            text-align: center;
            color: rgba(226,232,240,0.35);
            font-size: 0.78rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None
if "last_title" not in st.session_state:
    st.session_state.last_title = None

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎙️ Webinar Insight Studio</h1>
        <p>Turn dense webinar transcripts into crisp marketing takeaways —
        then chat with your entire knowledge base, powered by LangChain, FAISS
        and a local Ollama model. 100% private, nothing leaves your machine.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar — upload & ingest
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📤 Ingest a Webinar")
    st.caption("Upload a transcript PDF to summarize and index it.")

    uploaded_files = st.file_uploader(
        "Drop transcript PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    chain_type = st.selectbox(
        "Summarization strategy",
        options=["auto", "stuff", "map_reduce", "refine"],
        index=["auto", "stuff", "map_reduce", "refine"].index(config.SUMMARY_CHAIN_TYPE)
        if config.SUMMARY_CHAIN_TYPE in ["auto", "stuff", "map_reduce", "refine"]
        else 0,
        help=(
            "auto: stuff (single fast call) for short docs, map_reduce for "
            "long ones. stuff: best for short transcripts (few pages) — one "
            "call, fastest & most coherent. map_reduce: scales to 50+ page "
            "transcripts. refine: narrative pass, chunk by chunk."
        ),
    )

    n_selected = len(uploaded_files) if uploaded_files else 0
    process_label = f"⚡ Process {n_selected} Transcript{'s' if n_selected != 1 else ''}" if n_selected else "⚡ Process Transcript(s)"
    process_clicked = st.button(process_label, use_container_width=True, disabled=n_selected == 0)

    st.markdown("---")
    st.markdown("### 📊 Session Overview")
    n_webinars = len(read_summary_records())
    n_chunks = total_chunk_count()
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="stat-value">{n_webinars}</div>
                <div class="stat-label">Webinars Indexed</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{n_chunks}</div>
                <div class="stat-label">Chunks in FAISS</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    if not st.session_state.confirm_reset:
        if st.button("🗑️ Clear All Data", use_container_width=True, disabled=n_webinars == 0 and n_chunks == 0):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.warning("This deletes the FAISS index and all summaries. Are you sure?")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Yes, clear", use_container_width=True):
                reset_store()
                st.session_state.chat_history = []
                st.session_state.last_summary = None
                st.session_state.last_title = None
                st.session_state.confirm_reset = False
                st.rerun()
        with c2:
            if st.button("Cancel", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()

    st.markdown("---")
    st.markdown("### ⚙️ Model Settings")
    st.markdown(
        f"""
        <span class="pill pill-teal">LLM: {config.OLLAMA_CHAT_MODEL}</span>
        <span class="pill pill-blue">Embeddings: {config.EMBEDDING_PROVIDER}</span>
        <span class="pill pill-amber">Chunk: {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}</span>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Change these in `config.py` or via environment variables. "
        "Make sure the model above has been pulled with `ollama pull <model>`."
    )

    st.markdown("---")
    st.markdown("### 📚 Indexed Webinars")
    records = read_summary_records()
    if records:
        for r in records:
            st.markdown(f"- **{r['webinar_title']}**")
    else:
        st.caption("No webinars indexed yet.")

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
if process_clicked and uploaded_files:
    overall_status = st.status(
        f"Processing {len(uploaded_files)} transcript(s)…", expanded=True
    )
    processed_summaries = []  # (title, summary_text) for files that succeeded this run
    failures = []  # (filename, error) for files that failed this run

    try:
        for idx, up_file in enumerate(uploaded_files, start=1):
            overall_status.write(f"---\n**File {idx}/{len(uploaded_files)}: {up_file.name}**")
            try:
                overall_status.write("📥 Saving uploaded file…")
                saved_path = save_uploaded_file(up_file)

                overall_status.write("✂️ Loading PDF & splitting into chunks…")
                webinar_doc = load_and_chunk_pdf(saved_path)
                overall_status.write(
                    f"Found **{webinar_doc.page_count} pages** → **{len(webinar_doc.chunks)} chunks**."
                )

                overall_status.write("🧠 Indexing chunks into FAISS…")
                add_chunks(webinar_doc.chunks)

                overall_status.write(f"📝 Generating summary using **{chain_type}** strategy…")
                summary_text = summarize_chunks(webinar_doc.chunks, chain_type=chain_type)

                overall_status.write("💾 Persisting summary with metadata…")
                add_summary(
                    doc_id=webinar_doc.doc_id,
                    webinar_title=webinar_doc.title,
                    summary_text=summary_text,
                    source_file=webinar_doc.source_path,
                    uploaded_at=webinar_doc.uploaded_at,
                )

                processed_summaries.append((webinar_doc.title, summary_text))
                overall_status.write(f"✅ Done with **{up_file.name}**")
            except Exception as file_err:
                failures.append((up_file.name, str(file_err)))
                overall_status.write(f"❌ Failed on **{up_file.name}**: {file_err}")

        if processed_summaries:
            # Show the most recently processed file's summary in the main panel.
            st.session_state.last_title, st.session_state.last_summary = processed_summaries[-1]

        if failures:
            overall_status.update(
                label=f"⚠️ Finished with {len(failures)} failure(s) out of {len(uploaded_files)}",
                state="error",
            )
        else:
            overall_status.update(
                label=f"✅ All {len(uploaded_files)} transcript(s) processed successfully!",
                state="complete",
            )

        if failures:
            st.error(
                "Some files failed to process:\n\n"
                + "\n".join(f"- **{name}**: {err}" for name, err in failures)
                + "\n\nMake sure Ollama is running locally (`ollama serve`) and that "
                f"the `{config.OLLAMA_CHAT_MODEL}` model has been pulled "
                f"(`ollama pull {config.OLLAMA_CHAT_MODEL}`)."
            )
    except Exception as e:
        overall_status.update(label="❌ Processing failed", state="error")
        st.error(f"Something went wrong: {e}")
        with st.expander("Show technical details"):
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# Main layout — left: Summary + Inspector.  Right: permanent QA side panel.
# ---------------------------------------------------------------------------
main_col, chat_col = st.columns([1.7, 1], gap="medium")

with main_col:
    st.markdown('<div class="section-title">📝 Latest Summary</div>', unsafe_allow_html=True)
    if st.session_state.last_summary:
        st.markdown(f'<div class="meta-row">Webinar: <b>{st.session_state.last_title}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="summary-card">{st.session_state.last_summary}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="glass-card">Upload and process a transcript to see its summary here.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">📖 All Summaries</div>', unsafe_allow_html=True)
    records = read_summary_records()
    if records:
        for r in records:
            with st.expander(f"📄 {r['webinar_title']}"):
                st.markdown(f'<div class="meta-row">Uploaded: {r["uploaded_at"]}</div>', unsafe_allow_html=True)
                st.write(r["summary"])
    else:
        st.caption("No summaries stored yet.")

    st.markdown('<div class="section-title">🔍 FAISS Vector Store Inspector</div>', unsafe_allow_html=True)
    meta_records = inspect_metadata(limit=100)
    if meta_records:
        st.markdown(f'<div class="glass-card">Indexed chunks: <b>{len(meta_records)}</b></div>', unsafe_allow_html=True)
        titles = sorted({m.get("webinar_title", "Unknown") for m in meta_records})
        selected_title = st.selectbox("Filter by webinar", options=["All"] + titles)

        filtered = meta_records if selected_title == "All" else [
            m for m in meta_records if m.get("webinar_title") == selected_title
        ]

        for m in filtered[:30]:
            tag = "Summary" if m.get("type") == "summary" else f"Chunk #{m.get('chunk_index', '-')}"
            st.markdown(
                f"""
                <div class="glass-card">
                    <span class="pill pill-teal">{m.get('webinar_title','Unknown')}</span>
                    <span class="pill pill-amber">{tag}</span>
                    <div style="margin-top:0.5rem; color:#CBD5E1; font-size:0.85rem;">{m.get('preview','')}…</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="glass-card">No vectors indexed yet. Process a transcript to populate the store.</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Permanent chat / QA side panel
# ---------------------------------------------------------------------------
with chat_col:
    st.markdown('<div class="chat-panel-title">💬 Ask About Your Webinars</div>', unsafe_allow_html=True)

    chat_panel = st.container(height=560, border=True)

    with chat_panel:
        if not st.session_state.chat_history:
            st.caption("Ask something like: *\"What are the key UGC strategies mentioned?\"*")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for s in msg["sources"]:
                            st.caption(f"📄 {s.metadata.get('webinar_title', 'Unknown')} — {s.page_content[:180]}…")

        user_question = st.chat_input("Ask a question about your webinars…")

        if user_question:
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.markdown(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking through your transcripts…"):
                    try:
                        answer, sources = answer_question(user_question)
                    except Exception as e:
                        answer = (
                            f"⚠️ Couldn't reach the local LLM: {e}\n\n"
                            f"Make sure Ollama is running (`ollama serve`) and that "
                            f"`{config.OLLAMA_CHAT_MODEL}` is pulled "
                            f"(`ollama pull {config.OLLAMA_CHAT_MODEL}`)."
                        )
                        sources = []
                        with st.expander("Show technical details"):
                            st.code(traceback.format_exc())
                st.markdown(answer)
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.caption(f"📄 {s.metadata.get('webinar_title', 'Unknown')} — {s.page_content[:180]}…")

            st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})

st.markdown(
    f'<div class="footer-note">Webinar Insight Studio · Local-first with Ollama · Session started {datetime.now().strftime("%b %d, %Y")}</div>',
    unsafe_allow_html=True,
)
