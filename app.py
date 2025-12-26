"""
Streamlit chat app for local Ollama models with optional Wikipedia retrieval.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

import httpx
from sentence_transformers import CrossEncoder
import streamlit as st

# Constants
OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_SEARCH_LIMIT = 5
SEARCH_BACKEND_LIMIT = 20
DEFAULT_EXTRACT_CHARS = 1200
RERANK_TOP_K = 5
CHUNK_TOKEN_TARGET = 300
REQUEST_TIMEOUT = 30.0
DEFAULT_USER_AGENT = os.getenv("WIKI_USER_AGENT", "LocalWikiRAG/1.0 (contact: local)")

MODEL_OPTIONS = ["phi4:14b", "llama3.1:8b-instruct-q5_K_M"]
RETRIEVAL_MODES = ["Auto", "Always Wikipedia", "No Wikipedia"]

# Prompts
TOOL_GATING_PROMPT = (
    "You are a classifier deciding whether to call Wikipedia. "
    "Given a user question, respond with ONLY valid JSON matching this schema: "
    "{\"search\": <true|false>, \"query\": \"<short wikipedia query>\"}. "
    "Set search=true when the question asks for factual knowledge, definitions, people, places, events, dates, or when uncertain. "
    "Set search=false for chit-chat, opinions, or questions unrelated to factual lookup. "
    "Make the query concise and tailored to Wikipedia search."
)

ANSWER_SYSTEM_PROMPT = (
    "You are a local assistant. Use the SOURCES section when it is provided. "
    "Cite sources with [n] where n corresponds to the source number. "
    "If SOURCES is empty, answer normally and clearly state when you do not have Wikipedia matches."
)


def ollama_chat(model: str, messages: List[Dict[str, str]], timeout: float = REQUEST_TIMEOUT) -> str:
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        response = httpx.post(OLLAMA_CHAT_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Ollama error: {exc}")
        return ""


def _strip_html_snippet(snippet: str) -> str:
    return re.sub(r"<[^>]+>", "", snippet)


def refine_search_query(raw_query: str) -> str:
    cleaned = re.sub(r"\?+$", "", raw_query).strip()
    tokens = cleaned.split()
    if len(tokens) >= 3:
        quoted = f'"{cleaned}"'
        keywords = [tok for tok in tokens if len(tok) > 2][:6]
        keyword_block = " ".join(keywords)
        return f"{quoted} {keyword_block}".strip()
    return cleaned


def wiki_search(
    query: str, backend_limit: int = SEARCH_BACKEND_LIMIT, user_agent: str = DEFAULT_USER_AGENT
) -> List[Dict[str, Any]]:
    if not query:
        return []
    refined_query = refine_search_query(query)
    params = {
        "action": "query",
        "list": "search",
        "srsearch": refined_query,
        "format": "json",
        "srlimit": backend_limit,
        "srnamespace": 0,
        "srqiprofile": "classic_noboostlinks",
    }
    headers = {"User-Agent": user_agent}
    try:
        resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("query", {}).get("search", [])
    except Exception as exc:  # pylint: disable-broad-exception-caught
        st.error(f"Wikipedia search error: {exc}")
        return []


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_search_results(
    search_results: List[Dict[str, Any]], query: str, limit: int
) -> List[Dict[str, Any]]:
    if not search_results:
        return []

    reranker = get_reranker()
    pairs = [
        (query, f"{res.get('title', '')} {_strip_html_snippet(res.get('snippet', ''))}")
        for res in search_results
    ]
    scores = reranker.predict(pairs)
    rescored: List[Dict[str, Any]] = []
    for res, score in zip(search_results, scores):
        enriched = res.copy()
        enriched["neural_score"] = float(score)
        rescored.append(enriched)

    ranked = sorted(rescored, key=lambda item: item.get("neural_score", 0.0), reverse=True)
    return ranked[:limit]


def wiki_extracts(
    pageids: List[int], chars: int = DEFAULT_EXTRACT_CHARS, user_agent: str = DEFAULT_USER_AGENT
) -> Dict[int, Dict[str, Any]]:
    if not pageids:
        return {}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exchars": chars,
        "exintro": 1,
        "pageids": "|".join(str(pid) for pid in pageids),
        "format": "json",
    }
    headers = {"User-Agent": user_agent}
    try:
        resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        return {int(pid): info for pid, info in pages.items()}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Wikipedia extract error: {exc}")
        return {}


def chunk_text(text: str, target_words: int = CHUNK_TOKEN_TARGET) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks: List[str] = []
    for idx in range(0, len(words), target_words):
        chunk_words = words[idx : idx + target_words]
        chunks.append(" ".join(chunk_words))
    return chunks


def select_top_passages(
    extracts: Dict[int, Dict[str, Any]], query: str
) -> List[Tuple[int, str, str]]:
    candidates: List[Tuple[int, str, str]] = []
    for pageid, info in extracts.items():
        title = info.get("title", "Untitled")
        extract = info.get("extract", "")
        for chunk in chunk_text(extract):
            candidates.append((pageid, title, chunk))

    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, chunk) for _, _, chunk in candidates]
    scores = reranker.predict(pairs)
    scored_chunks = [
        (pid, title, chunk, float(score))
        for (pid, title, chunk), score in zip(candidates, scores)
    ]
    scored_chunks.sort(key=lambda item: item[3], reverse=True)
    best_by_page: Dict[int, Tuple[int, str, str, float]] = {}
    for pid, title, chunk, score in scored_chunks:
        if pid not in best_by_page:
            best_by_page[pid] = (pid, title, chunk, score)
    return [(pid, title, chunk) for pid, title, chunk, _ in best_by_page.values()]


def build_context(
    search_results: List[Dict[str, Any]], extracts: Dict[int, Dict[str, Any]], query: str
) -> Tuple[str, List[Dict[str, str]]]:
    sources: List[Dict[str, str]] = []
    lines: List[str] = []
    top_passages = select_top_passages(extracts, query)
    for pageid, title, passage in top_passages:
        if all(src.get("title") != title for src in sources):
            url = f"https://en.wikipedia.org/?curid={pageid}"
            sources.append({"title": title, "url": url})
        idx = next(
            (i for i, src in enumerate(sources, start=1) if src.get("title") == title),
            len(sources),
        )
        lines.append(f"[{idx}] {title}\n{passage}\n")
    context = "\n".join(lines)
    return context, sources


def decide_search(question: str, model: str) -> Tuple[bool, str]:
    messages = [
        {"role": "system", "content": TOOL_GATING_PROMPT},
        {"role": "user", "content": question},
    ]
    raw_response = ollama_chat(model, messages)
    try:
        parsed = json.loads(raw_response)
        query = str(parsed.get("query", question)).strip() or question
        return bool(parsed.get("search", False)), query
    except json.JSONDecodeError:
        return True, question


def format_user_with_sources(question: str, context: str) -> str:
    sources_block = context if context else "(none)"
    return f"Question: {question}\n\nSOURCES:\n{sources_block}"


def render_sources(sources: List[Dict[str, str]]) -> None:
    with st.expander("Retrieved Wikipedia sources"):
        if not sources:
            st.write("No sources were retrieved.")
            return
        for idx, src in enumerate(sources, start=1):
            st.markdown(f"**[{idx}] [{src['title']}]({src['url']})")


def main() -> None:
    st.set_page_config(page_title="Local Wiki RAG Chat", page_icon="📚")
    st.title("Local Wikipedia Chat with Ollama")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Choose model", options=MODEL_OPTIONS, index=0)
    with col2:
        retrieval_mode = st.selectbox("Retrieval mode", options=RETRIEVAL_MODES, index=0)
    with col3:
        user_agent = st.text_input("Wikipedia User-Agent", value=DEFAULT_USER_AGENT)
    max_sources = st.slider(
        "Max Wikipedia sources (reranked)", 3, RERANK_TOP_K, min(DEFAULT_SEARCH_LIMIT, RERANK_TOP_K), 1
    )

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status("Processing request", expanded=True) as status:
            status.write(f"Model: {model}")
            status.write(f"Retrieval mode: {retrieval_mode}")

            search_results: List[Dict[str, Any]] = []
            extracts: Dict[int, Dict[str, Any]] = {}
            context = ""
            sources_display: List[Dict[str, str]] = []
            query = prompt

            lexical_limit = min(SEARCH_BACKEND_LIMIT, max_sources * 4)

            if retrieval_mode == "Always Wikipedia":
                status.write("Retrieval: forcing Wikipedia search")
                _, query = decide_search(prompt, model)
                status.write(f"Search query: '{query}' (limit {lexical_limit})")
                search_results = wiki_search(query, backend_limit=lexical_limit, user_agent=user_agent)
            elif retrieval_mode == "Auto":
                status.write("Retrieval: deciding whether to search Wikipedia")
                should_search, query = decide_search(prompt, model)
                status.write(
                    "Auto decision: "
                    + (f"searching Wikipedia for '{query}'" if should_search else "no search needed")
                )
                if should_search:
                    search_results = wiki_search(query, backend_limit=lexical_limit, user_agent=user_agent)
            # No Wikipedia mode skips retrieval

            if search_results:
                status.write(f"Lexical results (BM25): {len(search_results)}")
                top_k = min(max_sources, RERANK_TOP_K)
                search_results = rerank_search_results(search_results, query, top_k)
                status.write(f"Neural reranked to top {len(search_results)}")
                pageids = [int(item.get("pageid", 0)) for item in search_results if item.get("pageid")]
                extracts = wiki_extracts(pageids, user_agent=user_agent)
                context, sources_display = build_context(search_results, extracts, query)
                status.write(f"Built context from {len(sources_display)} sources")
            else:
                status.write("Wikipedia search returned no results")

            render_sources(sources_display)

            answer_messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": format_user_with_sources(prompt, context)},
            ]
            assistant_reply = ollama_chat(model, answer_messages)
            if not assistant_reply:
                assistant_reply = "I could not generate a response at this time."
                status.write("Assistant returned an empty reply")
            else:
                status.write("Assistant response generated")
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
            st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
            status.update(label="Done", state="complete")


if __name__ == "__main__":
    main()
