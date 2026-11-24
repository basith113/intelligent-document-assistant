# Knowledge AI Assistant

Upload a PDF, TXT, or Markdown document, ask a question, and inspect the passages cited in the answer. The Python RAG pipeline uses OpenAI for embeddings and text generation, Chroma for local similarity search, FastAPI for the API, and Streamlit for the chat UI.

## Run locally

Use Python 3.12. From the project folder in PowerShell:

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
~~~

Set `OPENAI_API_KEY` in `.env` to a key with access to the configured models. Start the API and UI in separate PowerShell windows opened in the project folder:

~~~powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
~~~

~~~powershell
.\.venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py --server.address=127.0.0.1
~~~

Open [the chat UI](http://127.0.0.1:8501) or [the interactive API docs](http://127.0.0.1:8000/docs). The tokenizer downloads its vocabulary on first use if it has not been cached. A missing API key leaves health and document management available. Uploads and chat requests that need OpenAI report a configuration error.

## Try the samples

Upload `data/documents/hr-policy.md` and `data/documents/product-catalogue.txt` using the sidebar, then click **Index documents**.

1. Ask “How many annual leave days do full-time employees receive?” and open **Sources**.
2. Follow up with “How many can they carry over?”
3. Ask “What warranty comes with the Trail Bottle?”
4. Ask “What is the office street address?” to try a question the samples cannot answer.

## Docker

Create and configure `.env` as above, then run:

~~~powershell
docker compose up --build
~~~

Open the same local UI and API URLs. Compose binds both ports to loopback and stores indexed knowledge in the `knowledge_storage` volume. `docker compose down` stops the services and preserves that volume.

## Architecture

~~~mermaid
flowchart LR
    UI[Streamlit] -->|HTTP and SSE| API[FastAPI]
    API --> D[Document service]
    D --> L[Loader and token chunker]
    L --> E[Async OpenAI embeddings]
    E --> V[(Chroma and document registry)]
    API --> C[Chat service and session history]
    C --> R[Rewrite and retrieve]
    R --> E
    R --> V
    R --> G[OpenAI Responses API]
    G -->|Deltas and validated citation IDs| API
~~~

- `app/rag/loader.py` normalizes UTF-8 text or extracts the PDF text layer page by page. `chunker.py` prefers paragraph and sentence boundaries near the end of each token window and keeps overlap at Unicode boundaries.
- Ingestion hashes the exact file bytes to catch duplicates, including renamed copies. Chroma stores the chunk text, vector, and source metadata. A small SQLite registry publishes each document after all chunks are written and records pending writes or deletes for recovery. The original upload bytes are released after processing.
- Chat rewrites follow-ups with recent session history, embeds the standalone query, and filters cosine-distance results. The generator receives only the question and retrieved excerpts for answering. The application checks citation IDs against those excerpts and returns only sources cited in the final answer.
- The SSE endpoint emits provisional text deltas and a final answer with citation objects. Failed or interrupted generation produces no completed conversation turn. Session history is bounded in memory and clears on API restart. Chroma and the registry are persistent.
- Run one API worker per Chroma directory. This foundation is intended for local use with a shared corpus and no application-level authentication. OpenAI receives extracted text for embedding, and retrieved excerpts and questions for generation. Image-only PDFs require an existing OCR text layer. Correct citation IDs associate an answer with retrieved excerpts, while factual support should be checked against the displayed text.

## Configuration

Defaults appear in `.env.example`. Keep `.env` private.

| Setting | Default | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Empty | OpenAI API credentials |
| `OPENAI_MODEL` | `gpt-4.1-mini` | Response and query rewriting model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for document and question embeddings |
| `CHROMA_PATH` | `./storage/chroma_db` | Persistent vector and document registry directory |
| `COLLECTION_NAME` | `knowledge_base` | Corpus name. Use a new name and re-upload when changing embedding models |
| `CHUNK_SIZE` | `500` | Maximum tokens per window |
| `CHUNK_OVERLAP` | `100` | Target token overlap between windows |
| `TOP_K` | `5` | Maximum number of retrieved passages |
| `MAX_DISTANCE` | `0.8` | Inclusive cosine distance threshold. Lower values are stricter |
| `MAX_HISTORY_MESSAGES` | `10` | Maximum recent messages per session, keeping complete turns |
| `MAX_SESSIONS` | `500` | Maximum remembered conversations |
| `MAX_UPLOAD_SIZE_MB` | `20` | Maximum file size in MiB |
| `TEMPERATURE` | `0.2` | Answer generation temperature |
| `RAG_API_URL` | `http://127.0.0.1:8000` | Backend URL as seen by Streamlit |

## API examples

| Method | Path | Response |
| --- | --- | --- |
| GET | `/health` | `200` and `{"status":"ok"}` |
| POST | `/documents/upload` | `201` and indexed document metadata. Multipart field `file` |
| GET | `/documents` | `200` and an array of indexed document metadata |
| DELETE | `/documents/{document_id}` | `204` |
| POST | `/chat` | `200` and the answer, sources, and session ID |
| POST | `/chat/stream` | `200` and SSE events |
| DELETE | `/chat/sessions/{session_id}` | `204` |

~~~powershell
curl.exe -F "file=@data/documents/hr-policy.md" http://127.0.0.1:8000/documents/upload
$docs = Invoke-RestMethod http://127.0.0.1:8000/documents
$body = @{ question = "How many annual leave days do employees receive?" } | ConvertTo-Json
$chat = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/chat -ContentType application/json -Body $body
$chat | ConvertTo-Json -Depth 5
~~~

Example chat response (shortened excerpt):

~~~json
{
  "session_id": "c2af1f10-7e30-424b-846a-bdf5569d67a0",
  "answer": "Full-time employees receive 20 days of paid annual leave per calendar year [S1].",
  "sources": [{
    "citation": "S1",
    "document_id": "b559b8847284c0829f5ddf3317267dce6f083403d1243574fbd5a9c28ad10cbae",
    "filename": "hr-policy.md",
    "page": null,
    "chunk_id": "b559b8847284c0829f5ddf3317267dce6f083403d1243574fbd5a9c28ad10cbae:0",
    "distance": 0.23,
    "excerpt": "Full-time employees receive 20 days of paid annual leave each calendar year."
  }]
}
~~~

Reuse the returned `session_id` for follow-up requests. To inspect a stream in PowerShell:

~~~powershell
$followUp = @{ question = "How many can they carry over?"; session_id = $chat.session_id } | ConvertTo-Json -Compress
$file = New-TemporaryFile
Set-Content -LiteralPath $file.FullName -Value $followUp -Encoding ascii
curl.exe -N -H "Content-Type: application/json" --data-binary "@$($file.FullName)" http://127.0.0.1:8000/chat/stream
Remove-Item -LiteralPath $file.FullName

Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/chat/sessions/$($chat.session_id)"
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:8000/documents/$($docs[0].document_id)"
~~~

A `delta` event contains `{"text":"..."}`. A `done` event has the same schema as `POST /chat` and supplies the final, authoritative answer. An `error` event contains `status` and `detail`. Clients should discard provisional text if the stream closes without `done`. Invalid JSON or invalid fields may receive an ordinary HTTP error before the stream starts. Upload error codes include `409` for duplicate bytes, `413` for oversized files, `415` for unsupported extensions, and `422` for unreadable or empty documents.

## Tests

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
~~~

Tests use temporary Chroma stores and mock OpenAI calls. They cover loading, chunking, embedding batches, vector search, duplicate uploads, deletion, restart persistence, session follow-ups, citations, and streaming errors.
