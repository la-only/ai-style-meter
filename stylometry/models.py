"""Shared corpus models."""

from dataclasses import dataclass
from pathlib import Path

from razdel import tokenize  # type: ignore[import-untyped]


@dataclass(frozen=True)
class Document:
    """One indivisible corpus file."""

    source_file: str
    text: str


@dataclass(frozen=True)
class TextChunk:
    """A chunk that always retains its source-file lineage."""

    text: str
    source_file: str
    chunk_index: int


def load_documents(directory: Path) -> list[Document]:
    """Load UTF-8 text files in a stable order without modifying the corpus."""

    if not directory.is_dir():
        raise ValueError(f"Corpus directory does not exist: {directory}")
    paths = sorted(directory.glob("*.txt"), key=lambda path: path.name)
    if not paths:
        raise ValueError(f"Corpus directory contains no .txt files: {directory}")
    return [
        Document(source_file=path.name, text=path.read_text(encoding="utf-8"))
        for path in paths
    ]


def chunk_document(document: Document, chunk_words: int = 200) -> list[TextChunk]:
    """Split one document into non-overlapping chunks while preserving lineage."""

    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    words = [
        token
        for token in tokenize(document.text)
        if any(character.isalpha() for character in token.text)
    ]
    chunks: list[TextChunk] = []
    minimum_tail_words = max(1, chunk_words // 2)
    for chunk_index, start in enumerate(range(0, len(words), chunk_words)):
        group = words[start : start + chunk_words]
        if len(group) < minimum_tail_words:
            break
        text = document.text[group[0].start : group[-1].stop]
        chunks.append(
            TextChunk(
                text=text,
                source_file=document.source_file,
                chunk_index=chunk_index,
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document], chunk_words: int = 200
) -> list[TextChunk]:
    """Chunk documents independently; no chunk can cross a file boundary."""

    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, chunk_words=chunk_words)
    ]
