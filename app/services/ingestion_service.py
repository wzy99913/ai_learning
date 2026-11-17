"""Business workflow for putting one document into the knowledge base."""

from dataclasses import dataclass
from pathlib import Path

from app.rag.document_loader import load_document
from app.rag.embedding import embed_texts
from app.rag.text_splitter import split_text
from app.rag.vector_store import ChromaVectorStore


@dataclass(frozen=True)
class IngestionResult:
    source: str
    chunk_count: int
    stored_count: int


class IngestionService:
    def __init__(self, vector_store: ChromaVectorStore):
        self._vector_store = vector_store

    def ingest(self, file_path: Path, source: str) -> IngestionResult:
        """Load, split, embed, and store one supported document."""
        if not file_path.is_file():
            raise FileNotFoundError(f"文件不存在：{file_path}")

        text = load_document(file_path)
        if not text:
            raise ValueError("文档内容为空")

        chunks = split_text(text)
        embeddings = embed_texts(chunks)
        stored_count = self._vector_store.replace_document(
            chunks=chunks,
            embeddings=embeddings,
            source=source,
        )

        return IngestionResult(
            source=source,
            chunk_count=len(chunks),
            stored_count=stored_count,
        )
