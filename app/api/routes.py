"""Thin FastAPI routes: validation, service call, and HTTP error mapping only."""

import logging
import os
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    KnowledgeQueryRequest,
    KnowledgeQueryResponse,
    SourceResponse,
    UploadResponse,
)
from app.core.config import Settings
from app.rag.document_loader import SUPPORTED_SUFFIXES
from app.services.agent_service import AgentService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


logger = logging.getLogger(__name__)
MAX_UPLOAD_SIZE = 10 * 1024 * 1024


def create_api_router(
    settings: Settings,
    ingestion_service: IngestionService,
    rag_service: RAGService,
    agent_service: AgentService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "ai-learning"}

    @router.post("/knowledge/upload", response_model=UploadResponse)
    async def upload_knowledge_file(file: UploadFile = File(...)) -> UploadResponse:
        if not file.filename:
            raise HTTPException(status_code=400, detail="没有获取到文件名")

        source = Path(file.filename).name
        suffix = Path(source).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            raise HTTPException(status_code=400, detail="只支持 PDF 和 TXT 文件")

        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        # Keep the original suffix at the end so document_loader can identify
        # a temporary PDF/TXT by Path.suffix before it is renamed into place.
        temporary_path = settings.upload_dir / f".{uuid4().hex}.uploading{suffix}"
        final_path = settings.upload_dir / source

        try:
            content = await file.read(MAX_UPLOAD_SIZE + 1)
            if not content:
                raise HTTPException(status_code=400, detail="上传文件为空")
            if len(content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="文件不能超过 10 MB")

            temporary_path.write_bytes(content)
            result = ingestion_service.ingest(temporary_path, source=source)
            os.replace(temporary_path, final_path)

            return UploadResponse(
                filename=result.source,
                chunk_count=result.chunk_count,
                stored_count=result.stored_count,
                message="文件上传并入库成功",
            )
        except HTTPException:
            raise
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("文档入库失败")
            raise HTTPException(status_code=500, detail="文档入库失败") from exc
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
            await file.close()

    @router.post("/knowledge/query", response_model=KnowledgeQueryResponse)
    async def knowledge_query(request: KnowledgeQueryRequest) -> KnowledgeQueryResponse:
        try:
            result = await rag_service.answer(request.question, request.top_k)
            return KnowledgeQueryResponse(
                answer=result.answer,
                sources=[
                    SourceResponse(
                        source=chunk.source,
                        chunk_index=chunk.chunk_index,
                        distance=chunk.distance,
                    )
                    for chunk in result.sources
                ],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("知识库问答失败")
            raise HTTPException(status_code=500, detail="知识库问答失败") from exc

    @router.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        try:
            reply = await agent_service.chat(request.message, request.session_id)
            return ChatResponse(reply=reply)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Agent 对话失败")
            raise HTTPException(status_code=500, detail="Agent 对话失败") from exc

    return router
