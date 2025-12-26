"""
Streamlit chat app for local Ollama models with optional Wikipedia retrieval.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import httpx
import streamlit as st

# Constants
OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_SEARCH_LIMIT = 5
SEARCH_BACKEND_LIMIT = 50
DEFAULT_EXTRACT_CHARS = 1200
MAX_SOURCES_CAP = 50
RERANK_TOP_K = MAX_SOURCES_CAP
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
    "If SOURCES is empty, answer normally and clearly state when you do not have Wikipedia matches. "
    "Provide a thorough response roughly a page in length, elaborating on relevant context and details."
)

KEYWORD_QUERY_PROMPT = (
    "You are preparing a Wikipedia query. "
    "Return ONLY the core subject name needed to search for the topic on Wikipedia. "
    "Do not include attributes or extra descriptors—just the minimal subject phrase in lowercase."
)


def log_status(status: Any, message: str, started_at: float) -> float:
    elapsed = time.perf_counter() - started_at
    status.write(f"{message} ({elapsed:.2f}s)")
    return time.perf_counter()


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
        results = resp.json().get("query", {}).get("search", [])
        if results:
            return results
        # Fallback: try the raw query if the refined one yielded nothing.
        if refined_query != query:
            fallback_params = params.copy()
            fallback_params["srsearch"] = query
            resp = httpx.get(WIKI_API, params=fallback_params, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("query", {}).get("search", [])
        return results
    except Exception as exc:  # pylint: disable-broad-exception-caught
        st.error(f"Wikipedia search error: {exc}")
        return []


@st.cache_resource(show_spinner=False)
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


def build_context(
    search_results: List[Dict[str, Any]],
    extracts: Dict[int, Dict[str, Any]],
    max_sources: int,
    min_citations: int = 3,
) -> Tuple[str, List[Dict[str, str]]]:
    sources: List[Dict[str, str]] = []
    lines: List[str] = []
    target = min(len(search_results), max(max_sources, min_citations))
    for result in search_results:
        pageid = int(result.get("pageid", 0))
        info = extracts.get(pageid, {})
        title = info.get("title") or result.get("title") or "Untitled"
        extract = info.get("extract", "").strip()
        if not extract:
            extract = _strip_html_snippet(result.get("snippet", "")).strip()
        chunks = chunk_text(extract) if extract else []
        passage = chunks[0] if chunks else extract
        if not passage:
            continue
        url = f"https://en.wikipedia.org/?curid={pageid}" if pageid else ""
        sources.append({"title": title, "url": url or f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"})
        idx = len(sources)
        lines.append(f"[{idx}] {title}\n{passage}\n")
        if len(sources) >= target:
            break
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


def make_keyword_query(question: str, model: str) -> str:
    messages = [
        {"role": "system", "content": KEYWORD_QUERY_PROMPT},
        {"role": "user", "content": question},
    ]
    raw = ollama_chat(model, messages)
    cleaned = raw.strip().replace("\n", " ")
    tokens = cleaned.split()
    if tokens:
        core_tokens = tokens[:3]
        return " ".join(core_tokens)
    return question


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
        "Max Wikipedia sources (reranked)", 3, MAX_SOURCES_CAP, min(DEFAULT_SEARCH_LIMIT, MAX_SOURCES_CAP), 1
    )
    use_bm25 = st.checkbox("Use BM25 rerank", value=True)

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.status("Processing request", expanded=True) as status:
            step_timer = time.perf_counter()
            step_timer = log_status(status, f"Model: {model}", step_timer)
            step_timer = log_status(status, f"Retrieval mode: {retrieval_mode}", step_timer)

            search_results: List[Dict[str, Any]] = []
            extracts: Dict[int, Dict[str, Any]] = {}
            context = ""
            sources_display: List[Dict[str, str]] = []
            query = prompt

            lexical_limit = min(SEARCH_BACKEND_LIMIT, max_sources * 4)

            if retrieval_mode == "Always Wikipedia":
                step_timer = log_status(status, "Retrieval: forcing Wikipedia search", step_timer)
                decision_start = time.perf_counter()
                _, query = decide_search(prompt, model)
                keyword_query = make_keyword_query(query, model)
                step_timer = log_status(status, f"Keyword query: '{keyword_query}'", decision_start)
                search_start = time.perf_counter()
                search_results = wiki_search(
                    keyword_query, backend_limit=lexical_limit, user_agent=user_agent
                )
                step_timer = log_status(
                    status, f"Search query: '{keyword_query}' (limit {lexical_limit})", search_start
                )
            elif retrieval_mode == "Auto":
                step_timer = log_status(status, "Retrieval: deciding whether to search Wikipedia", step_timer)
                should_search, query = decide_search(prompt, model)
                step_timer = log_status(
                    status,
                    "Auto decision: "
                    + (f"searching Wikipedia for '{query}'" if should_search else "no search needed"),
                    step_timer,
                )
                if should_search:
                    keyword_start = time.perf_counter()
                    keyword_query = make_keyword_query(query, model)
                    step_timer = log_status(status, f"Keyword query: '{keyword_query}'", keyword_start)
                    search_start = time.perf_counter()
                    search_results = wiki_search(
                        keyword_query, backend_limit=lexical_limit, user_agent=user_agent
                    )
                    step_timer = log_status(
                        status, f"Search query: '{keyword_query}' (limit {lexical_limit})", search_start
                    )
            # No Wikipedia mode skips retrieval

            if search_results:
                step_timer = log_status(
                    status, f"Lexical results (BM25): {len(search_results)}", step_timer
                )
                selection_cap = min(len(search_results), max(max_sources, 3))
                search_results = search_results[:selection_cap]
                step_timer = log_status(status, f"Selected top {len(search_results)} results", step_timer)
                pageids = [int(item.get("pageid", 0)) for item in search_results if item.get("pageid")]
                extract_start = time.perf_counter()
                extracts = wiki_extracts(pageids, user_agent=user_agent)
                step_timer = log_status(
                    status, f"Fetched extracts for {len(pageids)} pages", extract_start
                )
                if use_bm25:
                    try:
                        bm25_start = time.perf_counter()
                        from rank_bm25 import BM25Okapi

                        docs = []
                        for result in search_results:
                            pageid = int(result.get("pageid", 0))
                            extract = extracts.get(pageid, {}).get("extract", "")
                            if not extract:
                                extract = _strip_html_snippet(result.get("snippet", ""))
                            docs.append(extract or result.get("title", ""))
                        tokenized_docs = [doc.lower().split() for doc in docs]
                        bm25 = BM25Okapi(tokenized_docs)
                        keyword_tokens = (keyword_query if 'keyword_query' in locals() else query).lower().split()
                        scores = bm25.get_scores(keyword_tokens)
                        scored = list(zip(search_results, scores))
                        search_results = [item for item, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)]
                        step_timer = log_status(status, "BM25 rerank applied", bm25_start)
                    except Exception as exc:  # pylint: disable=broad-exception-caught
                        step_timer = log_status(
                            status, f"BM25 rerank skipped due to error: {exc}", bm25_start
                        )
                context, sources_display = build_context(
                    search_results, extracts, max_sources=max_sources
                )
                step_timer = log_status(
                    status, f"Built context from {len(sources_display)} sources", step_timer
                )
            else:
                step_timer = log_status(status, "Wikipedia search returned no results", step_timer)

            render_sources(sources_display)

            answer_messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": format_user_with_sources(prompt, context)},
            ]
            answer_start = time.perf_counter()
            assistant_reply = ollama_chat(model, answer_messages)
            if not assistant_reply:
                assistant_reply = "I could not generate a response at this time."
                step_timer = log_status(status, "Assistant returned an empty reply", answer_start)
            else:
                step_timer = log_status(status, "Assistant response generated", answer_start)
            with st.chat_message("assistant"):
                st.markdown(assistant_reply)
            st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})
            status.update(label="Done", state="complete")


if __name__ == "__main__":
    main()
