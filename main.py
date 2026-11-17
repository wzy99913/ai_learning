"""Application composition root: create dependencies once and register routes."""

from fastapi import FastAPI

from app.api.routes import create_api_router
from app.core.config import get_settings
from app.rag.vector_store import ChromaVectorStore
from app.services.agent_service import AgentService
from app.services.ingestion_service import IngestionService
from app.services.rag_service import RAGService


settings = get_settings()
vector_store = ChromaVectorStore(settings)
ingestion_service = IngestionService(vector_store)
rag_service = RAGService(settings, vector_store)
agent_service = AgentService(settings, rag_service)


app = FastAPI(
    title="AI Learning: RAG and Agent",
    description="A learning project for document ingestion, RAG Q&A, and tool-calling Agent.",
    version="2.0.0",
)

app.include_router(
    create_api_router(
        settings=settings,
        ingestion_service=ingestion_service,
        rag_service=rag_service,
        agent_service=agent_service,
    )
)
