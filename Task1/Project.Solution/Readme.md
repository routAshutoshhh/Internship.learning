
## Put your description for the project here:

This project is an automated Webinar Transcript Summarizer & QA System built to help marketing teams quickly extract insights from long, multi-hour webinar recordings without having to sit through the entire transcript.

The app takes webinar transcript PDFs as input, breaks them down into manageable chunks, and indexes them into a FAISS vector store using local embeddings (HuggingFace's all-MiniLM-L6-v2). Once indexed, it generates a concise, marketing-focused summary of each webinar — covering the overview, key frameworks, and actionable takeaways — using a local LLM served through Ollama (Llama 3), so everything runs privately with no data leaving the machine.

Beyond summarization, the app also supports interactive question-answering: users can ask natural language questions about any of the ingested webinars and get answers grounded in the actual transcript content, along with the specific source excerpts the answer was based on.

The whole thing is wrapped in a Streamlit interface with:
- A sidebar to upload one or multiple transcript PDFs at once and kick off processing
- A live status log showing each step (chunking, indexing, summarizing)
- A dashboard to browse generated summaries and inspect what's currently stored in the FAISS index
- A permanent chat panel to ask questions and see cited sources
- Session stats and a one-click option to clear all indexed data and start fresh

Built with LangChain (for chaining prompts and retrieval), FAISS (for similarity search), and Ollama (for running the LLM locally), the goal was to make a genuinely useful, private, and fast tool for turning dense webinar content into something a marketing manager could actually skim and query in minutes instead of hours.

# Happy coding.
