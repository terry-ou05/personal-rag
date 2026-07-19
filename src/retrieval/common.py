from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
NO_PAGE_VALUE = "no_page"


@dataclass
class RetrievalResult:
    chunk_id: str
    page_content: str
    metadata: dict[str, Any]
    dense_score: float | None = None
    dense_rank: int | None = None
    bm25_score: float | None = None
    bm25_rank: int | None = None
    fused_score: float | None = None
    fused_rank: int | None = None

    def to_document(self) -> Document:
        metadata = dict(self.metadata)
        metadata["chunk_id"] = self.chunk_id
        if self.dense_rank is not None:
            metadata["dense_rank"] = self.dense_rank
        if self.dense_score is not None:
            metadata["dense_score"] = self.dense_score
        if self.bm25_rank is not None:
            metadata["bm25_rank"] = self.bm25_rank
        if self.bm25_score is not None:
            metadata["bm25_score"] = self.bm25_score
        if self.fused_rank is not None:
            metadata["fused_rank"] = self.fused_rank
        if self.fused_score is not None:
            metadata["fused_score"] = self.fused_score
        return Document(page_content=self.page_content, metadata=metadata)


def normalize_source_path(value: str | None) -> str:
    if value is None:
        return ""
    normalized = str(value).replace("\\", "/").strip()
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    return str(PurePosixPath(normalized))


def normalize_source_name(value: str | None) -> str:
    normalized = normalize_source_path(value)
    if not normalized:
        return ""
    return PurePosixPath(normalized).name


def normalize_page(value: Any) -> str:
    if value in (None, "", "-", "N/A"):
        return NO_PAGE_VALUE
    return str(value)


def build_chunk_id(metadata: dict[str, Any], chunk_index: int) -> str:
    source = normalize_source_path(str(metadata.get("source", "")))
    page = normalize_page(metadata.get("page"))
    return f"{source}::{page}::{chunk_index}"


def split_documents_with_chunk_ids(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    counters: dict[tuple[str, str], int] = {}

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata["source"] = normalize_source_path(metadata.get("source", ""))
        source = metadata["source"]
        page = normalize_page(metadata.get("page"))
        key = (source, page)
        chunk_index = counters.get(key, 0)
        counters[key] = chunk_index + 1
        metadata["chunk_index"] = chunk_index
        metadata["chunk_id"] = build_chunk_id(metadata, chunk_index)
        chunk.metadata = metadata

    return chunks


def metadata_matches_filter(metadata: dict[str, Any], metadata_filter: dict | None) -> bool:
    if not metadata_filter:
        return True

    if "$and" in metadata_filter:
        clauses = metadata_filter["$and"]
        if not isinstance(clauses, list):
            raise ValueError("metadata filter '$and' must be a list.")
        return all(metadata_matches_filter(metadata, clause) for clause in clauses)

    for field, expected_value in metadata_filter.items():
        if str(metadata.get(field, "")) != str(expected_value):
            return False
    return True


def result_from_document(
    document: Document,
    *,
    dense_score: float | None = None,
    dense_rank: int | None = None,
    bm25_score: float | None = None,
    bm25_rank: int | None = None,
) -> RetrievalResult:
    metadata = dict(document.metadata)
    chunk_id = str(metadata.get("chunk_id") or build_chunk_id(metadata, 0))
    metadata["chunk_id"] = chunk_id
    return RetrievalResult(
        chunk_id=chunk_id,
        page_content=document.page_content,
        metadata=metadata,
        dense_score=dense_score,
        dense_rank=dense_rank,
        bm25_score=bm25_score,
        bm25_rank=bm25_rank,
    )
