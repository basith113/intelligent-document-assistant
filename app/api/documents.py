"""Document upload, listing, and deletion endpoints."""

from typing import Annotated

from fastapi import APIRouter, File, Path, Request, Response, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.errors import AppError
from app.rag.loader import validate_filename
from app.schemas.document import Document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", status_code=201)
async def upload(request: Request, file: Annotated[UploadFile, File()]) -> Document:
    service = request.app.state.documents
    try:
        filename = validate_filename(file.filename or "")
        if file.size is not None and file.size > service.max_bytes:
            raise AppError(413, "The file exceeds the upload size limit.")
        content = await file.read(service.max_bytes + 1)
        return await service.ingest(filename, content)
    finally:
        await file.close()


@router.get("")
async def list_documents(request: Request) -> list[Document]:
    return await run_in_threadpool(request.app.state.documents.store.list_documents)


@router.delete("/{document_id}", status_code=204)
async def delete_document(
    request: Request, document_id: Annotated[str, Path(pattern=r"^[0-9a-f]{64}$")]
) -> Response:
    await run_in_threadpool(request.app.state.documents.store.delete, document_id)
    return Response(status_code=204)
