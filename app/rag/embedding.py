"""Lazy-loading embedding model used by both ingestion and retrieval."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache
def get_embedding_model() -> SentenceTransformer:
    """Load the cached model on first use without making a network request."""
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert document chunks into normalized vectors."""
    if not texts:
        raise ValueError("没有可向量化的文本")

    vectors = get_embedding_model().encode(
        texts,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Convert one user question into a normalized vector."""
    if not query.strip():
        raise ValueError("问题不能为空")

    vector = get_embedding_model().encode(
        query,
        normalize_embeddings=True,
    )
    return vector.tolist()
