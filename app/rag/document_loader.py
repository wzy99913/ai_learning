"""Load supported knowledge-base documents as plain text."""

from pathlib import Path

import pymupdf


SUPPORTED_SUFFIXES = {".pdf", ".txt"}


def load_document(file_path: Path) -> str:
    """Read a PDF or TXT file and return its text."""
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".txt":
        return _load_txt(file_path)

    raise ValueError("只支持 PDF 和 TXT 文件")


def _load_pdf(file_path: Path) -> str:
    pages: list[str] = []

    with pymupdf.open(file_path) as document:
        for page in document:
            pages.append(page.get_text())

    return "\n".join(pages).strip()


def _load_txt(file_path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return file_path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue

    raise ValueError("TXT 文件编码不支持，请使用 UTF-8 或 GB18030")
