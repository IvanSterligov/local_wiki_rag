"""
Streamlit chat app for local Ollama models with optional Wikipedia retrieval.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import httpx
import streamlit as st

# Constants
OLLAMA_URL = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT = f"{OLLAMA_URL}/api/chat"
WIKI_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_SEARCH_LIMIT = 5
DEFAULT_EXTRACT_CHARS = 4000
REQUEST_TIMEOUT = 60.0
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


def wiki_search(
    query: str,
    limit: int = DEFAULT_SEARCH_LIMIT,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = REQUEST_TIMEOUT,
) -> List[Dict[str, Any]]:
    if not query:
        return []
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit,
    }
    headers = {"User-Agent": user_agent}
    try:
        resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("query", {}).get("search", [])
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Wikipedia search error: {exc}")
        return []


def wiki_extracts(
    pageids: List[int],
    chars: int = DEFAULT_EXTRACT_CHARS,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = REQUEST_TIMEOUT,
) -> Dict[int, Dict[str, Any]]:
    if not pageids:
        return {}
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exchars": chars,
        "pageids": "|".join(str(pid) for pid in pageids),
        "format": "json",
    }
    headers = {"User-Agent": user_agent}
    try:
        resp = httpx.get(WIKI_API, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        return {int(pid): info for pid, info in pages.items()}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        st.error(f"Wikipedia extract error: {exc}")
        return {}


def build_context(search_results: List[Dict[str, Any]], extracts: Dict[int, Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    sources: List[Dict[str, str]] = []
    lines: List[str] = []
    for result in search_results:
        pageid = result.get("pageid")
        if pageid is None or int(pageid) not in extracts:
            continue
        extract = extracts[int(pageid)].get("extract", "")
        title = extracts[int(pageid)].get("title", result.get("title", "Untitled"))
        url = f"https://en.wikipedia.org/?curid={pageid}"
        sources.append({"title": title, "url": url})
        idx = len(sources)
        lines.append(f"[{idx}] {title} - {url}\n{extract}\n")
    context = "\n".join(lines)
    return context, sources


def decide_search(question: str, model: str, timeout: float) -> Tuple[bool, str]:
    messages = [
        {"role": "system", "content": TOOL_GATING_PROMPT},
        {"role": "user", "content": question},
    ]
    raw_response = ollama_chat(model, messages, timeout=timeout)
    try:
        parsed = json.loads(raw_response)
        return bool(parsed.get("search", False)), str(parsed.get("query", question))
    except json.JSONDecodeError:
        return False, question


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

    with st.expander("Advanced settings"):
        st.caption("Tune request timeouts if your local models respond slowly.")
        ollama_timeout = st.slider("Ollama timeout (seconds)", 10, 180, int(REQUEST_TIMEOUT))
        wiki_timeout = st.slider("Wikipedia timeout (seconds)", 5, 60, int(REQUEST_TIMEOUT / 2))

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        progress = st.progress(0)
        with st.status("Processing request...", expanded=True) as status:
            status.write("Starting request...")

            search_results: List[Dict[str, Any]] = []
            extracts: Dict[int, Dict[str, Any]] = {}
            context = ""
            sources_display: List[Dict[str, str]] = []

            if retrieval_mode == "Always Wikipedia":
                status.write("Retrieval mode: always use Wikipedia.")
                search_results = wiki_search(prompt, user_agent=user_agent, timeout=wiki_timeout)
            elif retrieval_mode == "Auto":
                status.write("Deciding whether Wikipedia lookup is needed...")
                should_search, query = decide_search(prompt, model, timeout=ollama_timeout)
                if should_search:
                    status.write(f"Model requested Wikipedia search for: {query}")
                    search_results = wiki_search(query, user_agent=user_agent, timeout=wiki_timeout)
                else:
                    status.write("Model skipped Wikipedia search for this turn.")
            else:
                status.write("Retrieval mode: No Wikipedia.")

            progress.progress(0.35)

            if search_results:
                status.write(f"Wikipedia search returned {len(search_results)} result(s). Fetching extracts...")
                pageids = [int(item.get("pageid", 0)) for item in search_results if item.get("pageid")]
                extracts = wiki_extracts(pageids, user_agent=user_agent, timeout=wiki_timeout)
                context, sources_display = build_context(search_results, extracts)
                status.write(f"Prepared context from {len(sources_display)} source(s).")
            else:
                status.write("No Wikipedia results were used.")

            progress.progress(0.6)
            render_sources(sources_display)

            answer_messages = [
                {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                {"role": "user", "content": format_user_with_sources(prompt, context)},
            ]
            status.write(f"Sending question to model '{model}'...")
            assistant_reply = ollama_chat(model, answer_messages, timeout=ollama_timeout)
            progress.progress(0.9)
            if not assistant_reply:
                assistant_reply = "I could not generate a response at this time."
                status.write("Model did not return a response.")
            else:
                status.write("Received model response.")

            status.update(label="Finished", state="complete")
            progress.progress(1.0)

        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_reply})


if __name__ == "__main__":
    main()
