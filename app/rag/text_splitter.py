"""Split long document text into overlapping chunks for vector retrieval."""


def split_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size, overlapping non-empty chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks
