from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.core.security import AuthenticatedUser, current_user, require_admin
from app.db.models import DocumentProcessingTask, KbChunk, KbDocument
from app.db.session import get_session
from app.schemas.common import ApiResponse, PageResult
from app.services.document_processing_service import document_processing_service

router = APIRouter(prefix="/admin/documents", tags=["documents"])
public_router = APIRouter(prefix="/knowledge/documents", tags=["knowledge-documents"])


def document_response(row: KbDocument) -> dict[str, object]:
    return {
        "id": row.id,
        "originalName": row.original_name,
        "storageName": row.storage_name,
        "fileType": row.file_type,
        "fileSize": row.file_size,
        "status": row.status,
        "chunkCount": row.chunk_count,
        "failureReason": row.failure_reason,
        "createdAt": row.created_at,
        "updatedAt": row.updated_at,
    }


def public_document_response(row: KbDocument) -> dict[str, object]:
    return {
        "id": row.id,
        "originalName": row.original_name,
        "fileType": row.file_type,
        "status": row.status,
        "chunkCount": row.chunk_count,
        "updatedAt": row.updated_at,
    }


@public_router.get("")
async def list_public_documents(
    _user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[list[dict[str, object]]]:
    rows = (
        (
            await session.execute(
                select(KbDocument)
                .where(KbDocument.status.in_(["READY", "COMPLETED"]))
                .order_by(KbDocument.updated_at.desc(), KbDocument.id.desc())
            )
        )
        .scalars()
        .all()
    )
    return ApiResponse.ok([public_document_response(row) for row in rows])


@public_router.get("/{document_id}")
async def get_public_document(
    document_id: int,
    _user: AuthenticatedUser = Depends(current_user),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[dict[str, object]]:
    document = await session.get(KbDocument, document_id)
    if document is None or document.status not in {"READY", "COMPLETED"}:
        raise NotFoundError("知识库文档不存在")
    chunks = (
        (
            await session.execute(
                select(KbChunk)
                .where(KbChunk.document_id == document.id)
                .order_by(KbChunk.chunk_index.asc())
            )
        )
        .scalars()
        .all()
    )
    return ApiResponse.ok(
        {
            **public_document_response(document),
            "content": "\n\n".join(chunk.content for chunk in chunks),
            "chunks": [
                {"id": chunk.id, "chunkIndex": chunk.chunk_index, "content": chunk.content}
                for chunk in chunks
            ],
        }
    )


@router.get("")
async def list_documents(
    page: int = Query(default=1, ge=1),
    size: int = Query(default=20, ge=1, le=100),
    keyword: str | None = None,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[PageResult[dict[str, object]]]:
    query = select(KbDocument)
    count_query = select(func.count()).select_from(KbDocument)
    if keyword:
        like = f"%{keyword}%"
        query = query.where(KbDocument.original_name.like(like))
        count_query = count_query.where(KbDocument.original_name.like(like))
    total = int((await session.execute(count_query)).scalar_one())
    rows = (
        (await session.execute(query.order_by(KbDocument.created_at.desc()).offset((page - 1) * size).limit(size)))
        .scalars()
        .all()
    )
    return ApiResponse.ok(
        PageResult(page=page, size=size, total=total, records=[document_response(row) for row in rows])
    )


@router.post("")
async def upload_document(
    file: UploadFile = File(...),
    admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[dict[str, object]]:
    content = await file.read()
    row = await document_processing_service.save_upload(
        session,
        file.filename or "unnamed.txt",
        content,
        admin.user_id,
    )
    return ApiResponse.ok(document_response(row))


@router.post("/{document_id}/retry")
async def retry_document(
    document_id: int,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[None]:
    await document_processing_service.retry(session, document_id)
    return ApiResponse.ok(None)


@router.post("/{document_id}/process")
async def process_document(
    document_id: int,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[dict[str, object]]:
    row = await session.get(KbDocument, document_id)
    if row is None:
        raise NotFoundError("文档不存在")
    await document_processing_service.retry(session, document_id)
    task = (
        await session.execute(select(DocumentProcessingTask).where(DocumentProcessingTask.document_id == document_id))
    ).scalar_one()
    await document_processing_service.process_task(session, task.id)
    await session.refresh(row)
    return ApiResponse.ok({"processed": True, "document": document_response(row)})


@router.get("/{document_id}/download")
async def download_document(
    document_id: int,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    row = await session.get(KbDocument, document_id)
    if row is None:
        raise NotFoundError("文档不存在")
    path = Path(row.storage_path)
    if not path.exists() or not path.is_file():
        raise NotFoundError("文档文件不存在")
    return FileResponse(path, filename=row.original_name, media_type="application/octet-stream")


@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    _admin: AuthenticatedUser = Depends(require_admin),
    session: AsyncSession = Depends(get_session),
) -> ApiResponse[None]:
    await document_processing_service.delete(session, document_id)
    return ApiResponse.ok(None)
