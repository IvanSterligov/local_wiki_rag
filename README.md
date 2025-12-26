# Local Wikipedia RAG Chat

A small Streamlit chat app that talks to local Ollama models and can optionally retrieve English Wikipedia content for grounding.

## Features
- Switch between local Ollama models (`phi4:14b`, `llama3.1:8b-instruct-q5_K_M`).
- Tool-gated Wikipedia retrieval with Auto/Always/Never modes.
- MediaWiki API search + extracts with source previews and required User-Agent header.
- Chat history persistence in the UI and citations aligned with provided sources.
- In-app progress + status logging plus adjustable Ollama/Wikipedia timeouts to diagnose slow responses.

## Requirements
- Python 3.9+
- Ollama running locally with the desired models pulled.
- Network access to `https://en.wikipedia.org/w/api.php`.
- Verify your installed tags with `ollama list`; update `app.py` if your tag names differ.

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
- Use the **Advanced settings** expander to raise timeouts if your local models take longer to answer and to watch per-step status updates during each turn.
