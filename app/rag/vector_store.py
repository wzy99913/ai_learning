"""A focused ChromaDB adapter for storing and retrieving document chunks."""

import hashlib
from dataclasses import dataclass

import chromadb

from app.core.config import Settings


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    chunk_index: int
    distance: float


class ChromaVectorStore:
    """Own all direct ChromaDB calls for one knowledge-base collection."""

    def __init__(self, settings: Settings):
        self._client = chromadb.PersistentClient(path=str(settings.chroma_db_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def replace_document(
        self,
        chunks: list[str],
        embeddings: list[list[float]],
        source: str,
    ) -> int:
        """Replace all chunks from one source, preventing duplicate uploads."""
        if len(chunks) != len(embeddings):
            raise ValueError("文本块数量与向量数量不一致")
        if not chunks:
            raise ValueError("没有可保存的文本块")

        existing = self._collection.get(where={"source": source})
        existing_ids = existing.get("ids", [])
        if existing_ids:
            self._collection.delete(ids=existing_ids)

        ids = [self._make_chunk_id(source, index) for index in range(len(chunks))]
        metadatas = [
            {"source": source, "chunk_index": index}
            for index in range(len(chunks))
        ]

        self._collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Return up to top_k chunks closest to the query embedding."""
        total = self._collection.count()
        if total == 0:
            return []

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, total),
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        return [
            RetrievedChunk(
                text=document,
                source=metadata["source"],
                chunk_index=metadata["chunk_index"],
                distance=distance,
            )
            for document, metadata, distance in zip(documents, metadatas, distances)
        ]

    def count(self) -> int:
        return self._collection.count()

    @staticmethod
    def _make_chunk_id(source: str, chunk_index: int) -> str:
        raw_id = f"{source}_{chunk_index}"
        return hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
