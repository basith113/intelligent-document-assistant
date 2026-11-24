"""Streamlit interface for the Knowledge AI Assistant REST API."""

import json
import os
from collections.abc import Iterator
from uuid import uuid4

import httpx
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = httpx.Timeout(300.0, connect=5.0)


class APIError(Exception):
    pass


def check(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.read()
        message = json.loads(detail).get("detail")
    except (ValueError, AttributeError):
        message = None
    raise APIError(message if isinstance(message, str) else f"API request failed ({response.status_code}).")


def request(method: str, path: str, **kwargs) -> httpx.Response:
    response = httpx.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    check(response)
    return response


def events(response: httpx.Response) -> Iterator[tuple[str, dict]]:
    """Parse SSE event names and data, including multiline events."""
    name = "message"
    data = []
    for line in response.iter_lines():
        if line.startswith("event:"):
            name = line[6:].lstrip(" ")
        elif line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line:
            if data:
                yield name, json.loads("\n".join(data))
            name, data = "message", []


def stream_answer(question: str) -> Iterator[tuple[str, dict]]:
    with httpx.stream(
        "POST", f"{API_URL}/chat/stream",
        json={"session_id": st.session_state.session_id, "question": question},
        timeout=TIMEOUT,
    ) as response:
        check(response)
        yield from events(response)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            page = f" · page {source['page']}" if source["page"] else ""
            st.text(f"[{source['citation']}] {source['filename']}{page}")
            st.caption(f"Cosine distance {source['distance']:.3f} · {source['chunk_id']}")
            st.text(source["excerpt"])


st.set_page_config(page_title="Knowledge AI Assistant", page_icon="📚", layout="centered")
st.title("Knowledge AI Assistant")
st.caption("Ask questions about your uploaded documents and inspect the cited passages.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []
    st.session_state.upload_key = 0

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader(
        "Upload PDF, TXT, or Markdown",
        type=["pdf", "txt", "md", "markdown"],
        accept_multiple_files=True,
        key=f"upload_{st.session_state.upload_key}",
    )
    if st.button("Index documents", disabled=not files, type="primary"):
        has_error = False
        with st.spinner("Extracting and indexing documents…"):
            for file in files:
                try:
                    document = request(
                        "POST", "/documents/upload", files={"file": (file.name, file.getvalue())}
                    ).json()
                    st.success(f"Indexed {document['filename']} ({document['chunk_count']} chunks).")
                except (APIError, httpx.HTTPError) as error:
                    has_error = True
                    st.error(f"{file.name}: {error}")
        if not has_error:
            st.session_state.upload_key += 1
            st.toast("Documents indexed.")
            st.rerun()

    try:
        documents = request("GET", "/documents").json()
        if not documents:
            st.caption("No documents indexed yet.")
        for document in documents:
            info, action = st.columns([4, 2], vertical_alignment="center")
            info.text(document["filename"])
            info.caption(f"{document['chunk_count']} chunks")
            if action.button("Delete", key=f"delete_{document['document_id']}"):
                request("DELETE", f"/documents/{document['document_id']}")
                st.rerun()
    except (APIError, httpx.HTTPError) as error:
        st.error(f"Could not load or update documents: {error}")

    st.divider()
    if st.button("Clear conversation"):
        try:
            request("DELETE", f"/chat/sessions/{st.session_state.session_id}")
        except (APIError, httpx.HTTPError) as error:
            st.error(f"Could not clear the conversation: {error}")
        else:
            st.session_state.session_id = str(uuid4())
            st.session_state.messages.clear()
            st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

if question := st.chat_input("Ask about your documents", max_chars=2000):
    question = question.strip()
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            slot = st.empty()
            slot.caption("Searching documents…")
            parts = []
            result = None
            try:
                for event, payload in stream_answer(question):
                    if event == "delta":
                        parts.append(payload["text"])
                        slot.markdown("".join(parts) + " ▌")
                    elif event == "done":
                        result = payload
                    elif event == "error":
                        raise APIError(payload["detail"])
                if result is None:
                    raise APIError("The connection ended before the answer was complete. Try again.")
            except (APIError, httpx.HTTPError, ValueError, KeyError) as error:
                slot.error(f"Could not complete the answer: {error}")
            else:
                slot.markdown(result["answer"])
                render_sources(result["sources"])
                st.session_state.messages.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": result["answer"], "sources": result["sources"]},
                ])
