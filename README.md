# Local RAG System

A completely offline, secure, and privacy-first Retrieval-Augmented Generation (RAG) system built with **Python**, **LangChain**, and **Streamlit**. It leverages local embeddings and the Ollama ecosystem to serve highly accurate document-based QA without relying on external cloud APIs.

## Architecture Overview

The system architecture is designed to be fully isolated and local. Here is the technical stack:

- **Frontend Interface:** [Streamlit](https://streamlit.io/) handles the interactive web UI, real-time token streaming, and dynamic file uploads.
- **LLM Backend:** Local LLaMA 3.2 running via [Ollama](https://ollama.com/). The model is instantiated with zero temperature (`temperature=0.0`) and an expanded context window (`num_ctx=4096`) to ensure strict factual precision and maximum context retention.
- **Embeddings:** HuggingFace's `all-MiniLM-L6-v2` handles vector transformations for semantic similarity.
- **Vector Storage:** [ChromaDB](https://www.trychroma.com/) runs persistently on disk (`./chroma_db`).
- **Data Pipelines:** PyPDFLoader and TextLoader power the ingestion engine, actively chunking documents via a `SemanticChunker` rather than arbitrary character limits. This advanced strategy utilizes local HuggingFace embeddings to determine splitting thresholds dynamically. It groups sentences by calculating semantic similarity variations, triggering hard chunk boundaries only when similarity scores drop past a strict 90th-percentile breakpoint ceiling.

## Key Features

1. **Multi-User Session Isolation:** Streamlit assigns a unique UUID to every session. ChromaDB automatically spins up dedicated, ephemeral collections isolated to that specific user ID to ensure zero cross-contamination of documents between active users.
2. **History-Aware Memory:** Utilizes LangChain's `create_history_aware_retriever`. The chatbot tracks your conversation, allowing you to ask ambiguous follow-up questions referencing past context.
3. **Pre-Retrieval Guardrails:** An explicit local router agent analyzes every query before vector search begins. If a question is entirely off-topic (e.g., asking for recipes or general coding help), the router halts execution and returns a strict refusal, preventing hallucinations.
4. **Real-Time Streaming:** The LLM output is streamed natively to the Streamlit UI, providing instant feedback without blocking the main thread.
5. **Source Attribution:** Responses cleanly aggregate and render the specific source files and page numbers used to generate the context.

## Getting Started

### 1. Prerequisites
- Python 3.10+ (Note: Python 3.9 is out-of-date and incompatible with modern LangChain v1 structural baselines)
- [Ollama](https://ollama.com/download) installed locally.
- LLaMA 3.2 pulled locally:
  ```bash
  ollama run llama3.2
  ```

### 2. Installation
Clone the repository and set up a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pre-Cache HuggingFace Models (For Offline Support)
Before migrating to an air-gapped environment without internet access, you must securely pre-cache the local embedding transformers and cross-encoder models. While still connected to a network, run the Python cache command:

```bash
python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```
This forces HuggingFace to pre-download the requisite topological weights securely onto your local disk.

### 4. Running the Application
Ensure the Ollama service is running on your machine, then launch the Streamlit frontend.

```bash
# We disable the file watcher to prevent dependency scanning conflicts (e.g., torchvision)
streamlit run ui.py --server.fileWatcherType none
```

Navigate to `http://localhost:8501` in your browser.

## Project Structure

```
├── .venv/                  # Virtual environment
├── data/                   # Temporary directory for uploaded documents
├── chroma_db/              # Persistent SQLite local vector storage
├── app.py                  # Core logic and CLI-based RAG testing
├── ui.py                   # Primary Streamlit web application
└── README.md               # Architecture and setup documentation
```

## Privacy & Security

This system is completely air-gapped from third-party telemetry. No API keys are required, and no document data is sent across the internet. All vectors and generation happen locally on your hardware.
