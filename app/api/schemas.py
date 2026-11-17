"""Pydantic request and response contracts for the HTTP API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    session_id: str = Field(min_length=1, max_length=100)


class ChatResponse(BaseModel):
    reply: str


class KnowledgeQueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10)


class SourceResponse(BaseModel):
    source: str
    chunk_index: int
    distance: float


class KnowledgeQueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class UploadResponse(BaseModel):
    filename: str
    chunk_count: int
    stored_count: int
    message: str
