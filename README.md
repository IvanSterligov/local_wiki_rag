# Local Wikipedia RAG Chat

A small Streamlit chat app that talks to local Ollama models and can optionally retrieve English Wikipedia content for grounding.

## Features
- Switch between local Ollama models (`phi4:14b`, `llama3.1:8b-instruct-q5_K_M`).
- Tool-gated Wikipedia retrieval with Auto/Always/Never modes.
- MediaWiki Search API with article-only filtering, CirrusSearch-style query refinement, and a two-stage BM25 → neural rerank pipeline.
- Passage-level extracts (lead sections, chunked) with source previews and required User-Agent header.
- Chat history persistence in the UI and citations aligned with provided sources.

## Requirements
- Python 3.9+
- Ollama running locally with the desired models pulled.
- Network access to `https://en.wikipedia.org/w/api.php`.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment variables
- `WIKI_USER_AGENT`: Optional string to identify your client to Wikipedia. Defaults to `LocalWikiRAG/1.0 (contact: local)`.

## Running the app

### Linux/macOS
```bash
export WIKI_USER_AGENT="LocalWikiRAG/1.0 (contact: you@example.com)"
streamlit run app.py
```

### Windows (PowerShell)
```powershell
$env:WIKI_USER_AGENT="LocalWikiRAG/1.0 (contact: you@example.com)"
streamlit run app.py
```

Open the displayed local URL in your browser. Select a model and retrieval mode, then chat.

## Notes
- Ollama API endpoint defaults to `http://localhost:11434/api/chat`; adjust constants in `app.py` if your setup differs.
- Only Wikipedia is queried for retrieval. The app continues gracefully if Wikipedia or Ollama requests fail.
- Wikipedia retrieval uses `srnamespace=0` with tightened queries, reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and
  lead-section passages to reduce prompt bloat.
