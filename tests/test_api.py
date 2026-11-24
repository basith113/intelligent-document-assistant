"""Exercise API contracts with real storage and mocked OpenAI HTTP responses."""

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import AsyncOpenAI

from app.api.chat import EventStreamResponse
from app.core.config import Settings
from app.main import create_app
from app.rag.prompts import INSUFFICIENT


def mock_response(text):
    return {
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }],
    }


@pytest.fixture
def api(tmp_path):
    state = {"calls": [], "answer": "Employees receive 20 days [S1].", "interrupt": False, "rate_limit": False}

    def openai(request):
        body = json.loads(request.content)
        state["calls"].append(body)
        if state["rate_limit"]:
            return httpx.Response(429, json={"error": {"message": "mock quota"}})
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(200, json={"data": [
                {"index": index, "embedding": [0.0, 1.0] if "weather" in text else [1.0, 0.0]}
                for index, text in enumerate(body["input"])
            ]})
        if "Rewrite" in body["instructions"]:
            return httpx.Response(200, json=mock_response("How many days of annual leave?"))
        if body.get("stream"):
            events = [
                {"type": "response.output_text.delta", "delta": state["answer"][:15]},
                {"type": "response.output_text.delta", "delta": state["answer"][15:]},
            ]
            if not state["interrupt"]:
                events.append({"type": "response.completed", "response": mock_response(state["answer"])})
            return httpx.Response(200, text="".join(f"data: {json.dumps(event)}\n\n" for event in events))
        return httpx.Response(200, json=mock_response(state["answer"]))

    client = AsyncOpenAI(
        api_key="test-key", max_retries=0, http_client=httpx.AsyncClient(transport=httpx.MockTransport(openai))
    )
    settings = Settings(chroma_path=tmp_path, max_upload_size_mb=1, max_distance=0.25)
    with TestClient(create_app(settings, client)) as test_client:
        yield test_client, state


def upload(client, name="policy.md", data=b"Employees receive 20 days of annual leave."):
    return client.post("/documents/upload", files={"file": (name, data)})


def test_document_api(api):
    client, provider = api
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/documents").json() == []
    assert upload(client, "binary.zip").status_code == 415
    assert upload(client, data=b"x" * 1_048_577).status_code == 413
    assert upload(client, data=b"").status_code == 422
    response = upload(client)
    assert response.status_code == 201
    document = response.json()
    assert upload(client, "copy.txt").status_code == 409
    assert len(provider["calls"]) == 1
    assert client.get("/documents").json() == [document]
    path = f"/documents/{document['document_id']}"
    assert client.delete(path).status_code == 204
    assert client.delete(path).status_code == 404
    assert client.get("/documents").json() == []
    assert upload(client).status_code == 201


def test_answer_followup_and_clear(api):
    client, provider = api
    upload(client)
    first = client.post("/chat", json={"question": "What is the leave allowance?"}).json()
    assert first["answer"] == provider["answer"]
    assert first["sources"][0]["filename"] == "policy.md"
    assert first["sources"][0]["page"] is None
    session_id = first["session_id"]
    question = {"question": "How many days is that?", "session_id": session_id}
    assert client.post("/chat", json=question).status_code == 200
    assert "Rewrite" in provider["calls"][-3]["instructions"]
    assert provider["calls"][-2]["input"] == ["How many days of annual leave?"]
    assert json.loads(provider["calls"][-1]["input"])["question"] == "How many days of annual leave?"
    assert client.delete(f"/chat/sessions/{session_id}").status_code == 204
    assert client.post("/chat", json=question).status_code == 200
    assert provider["calls"][-2]["input"] == [question["question"]]


def test_insufficient_context_and_validation(api):
    client, provider = api
    assert client.post("/chat", json={"question": " "}).status_code == 422
    assert client.post("/chat", json={"question": "weather"}).json()["answer"] == INSUFFICIENT
    assert provider["calls"] == []
    upload(client)
    answer = client.post("/chat", json={"question": "weather"}).json()
    assert answer["answer"] == INSUFFICIENT
    assert answer["sources"] == []
    assert len(provider["calls"]) == 2
    provider["answer"] = "Unsupported claim [S42]."
    assert client.post("/chat", json={"question": "Leave?"}).json()["answer"] == INSUFFICIENT


def test_sse_and_interrupted_generation(api):
    client, provider = api
    upload(client)
    response = client.post("/chat/stream", json={"question": "Leave?"})
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.text.count("event: delta") == 2
    done = json.loads(response.text.split("event: done\ndata: ", 1)[1].strip())
    assert done["answer"] == provider["answer"]
    assert done["sources"][0]["citation"] == "S1"
    provider["interrupt"] = True
    request = {"question": "Leave?", "session_id": done["session_id"]}
    response = client.post("/chat/stream", json=request)
    assert "event: error" in response.text
    assert "event: done" not in response.text
    assert client.post("/chat", json=request).status_code == 200
    history = json.loads(provider["calls"][-3]["input"])["history"]
    assert len(history) == 2


def test_rate_limit_is_reported_without_indexing(api):
    client, provider = api
    provider["rate_limit"] = True
    assert upload(client).status_code == 429
    assert client.get("/documents").json() == []
    provider["rate_limit"] = False
    assert upload(client).status_code == 201


def test_missing_api_key_is_reported(tmp_path):
    with TestClient(create_app(Settings(chroma_path=tmp_path, openai_api_key=""))) as client:
        assert client.get("/health").status_code == 200
        assert upload(client).status_code == 503


def test_sse_closes_generator_on_client_disconnect():
    async def disconnect_during_send():
        disconnected = asyncio.Event()
        closed = asyncio.Event()

        async def events():
            try:
                yield "event: delta\n\n"
            finally:
                await asyncio.sleep(0)
                closed.set()

        async def receive():
            await disconnected.wait()
            return {"type": "http.disconnect"}

        async def send(message):
            if message["type"] == "http.response.body":
                disconnected.set()
                await asyncio.Future()

        scope = {"type": "http", "asgi": {"spec_version": "2.3"}}
        await asyncio.wait_for(EventStreamResponse(events())(scope, receive, send), timeout=2)
        assert closed.is_set()

    asyncio.run(disconnect_during_send())
