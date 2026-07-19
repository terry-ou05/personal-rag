import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.evaluate_hybrid_retrieval import build_v8_report
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.common import (
    RetrievalResult,
    build_chunk_id,
    metadata_matches_filter,
    normalize_source_name,
    normalize_source_path,
    split_documents_with_chunk_ids,
)
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.rrf import reciprocal_rank_fusion
from src.retrieval.tokenizer import tokenize


def make_doc(text: str, source: str, **metadata):
    values = {"source": source}
    values.update(metadata)
    return Document(page_content=text, metadata=values)


def test_tokenizer_preserves_it_tokens_and_chinese_terms():
    tokens = tokenize(
        "Nginx 502, df -h, error.log, max_connections, redis-server, "
        "OOM, 90%, /var/log, 127.0.0.1，服务器登录失败"
    )

    assert "nginx" in tokens
    assert "502" in tokens
    assert "df" in tokens
    assert "-h" in tokens
    assert "error.log" in tokens
    assert "max_connections" in tokens
    assert "redis-server" in tokens
    assert "oom" in tokens
    assert "90%" in tokens
    assert "/var/log" in tokens
    assert "127.0.0.1" in tokens
    assert any("登录" in token for token in tokens)
    assert tokenize("") == []


def test_chunk_id_is_stable_and_path_normalized():
    windows_metadata = {"source": "data\\raw\\ops.md", "page": 2}
    posix_metadata = {"source": "data/raw/ops.md", "page": 2}

    assert normalize_source_path(windows_metadata["source"]) == "data/raw/ops.md"
    assert normalize_source_name(windows_metadata["source"]) == "ops.md"
    assert build_chunk_id(windows_metadata, 0) == build_chunk_id(posix_metadata, 0)
    assert build_chunk_id(windows_metadata, 0) != build_chunk_id(
        {"source": "data/raw/other.md", "page": 2},
        0,
    )


def test_split_documents_adds_chunk_metadata():
    chunks = split_documents_with_chunk_ids(
        [make_doc("hello world", "data/raw/a.md", system="server")]
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["chunk_id"] == "data/raw/a.md::no_page::0"
    assert chunks[0].metadata["system"] == "server"


def test_metadata_filter_matches_chroma_style_and_clause():
    metadata = {"system": "nginx", "severity": "high"}

    assert metadata_matches_filter(metadata, {"system": "nginx"})
    assert metadata_matches_filter(
        metadata,
        {"$and": [{"system": "nginx"}, {"severity": "high"}]},
    )
    assert not metadata_matches_filter(metadata, {"system": "mysql"})


def test_bm25_retriever_basic_filter_and_empty_cases():
    docs = split_documents_with_chunk_ids(
        [
            make_doc("Nginx 502 Bad Gateway error.log upstream", "data/raw/nginx.md", system="nginx"),
            make_doc("Redis maxmemory OOM redis-server", "data/raw/redis.md", system="redis"),
        ]
    )
    retriever = BM25Retriever(docs)

    results = retriever.retrieve("502 error.log", top_k=1)
    assert results[0].metadata["source"] == "data/raw/nginx.md"
    assert results[0].bm25_rank == 1
    assert results[0].bm25_score is not None

    filtered = retriever.retrieve("redis OOM", top_k=2, metadata_filter={"system": "redis"})
    assert len(filtered) == 1
    assert filtered[0].metadata["system"] == "redis"

    assert retriever.retrieve("502 error.log", metadata_filter={"system": "mysql"}) == []
    with pytest.raises(ValueError, match="empty query"):
        retriever.retrieve("")
    with pytest.raises(ValueError, match="empty corpus"):
        BM25Retriever([])


def test_rrf_fuses_duplicate_and_single_route_results_with_stable_order():
    dense = [
        RetrievalResult("a", "doc a", {"source": "a.md"}, dense_score=0.1, dense_rank=1),
        RetrievalResult("b", "doc b", {"source": "b.md"}, dense_score=0.2, dense_rank=2),
    ]
    bm25 = [
        RetrievalResult("b", "doc b", {"source": "b.md"}, bm25_score=3.0, bm25_rank=1),
        RetrievalResult("c", "doc c", {"source": "c.md"}, bm25_score=2.0, bm25_rank=2),
    ]

    fused = reciprocal_rank_fusion(dense, bm25, final_top_k=3, rrf_k=60)

    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].dense_rank == 2
    assert fused[0].bm25_rank == 1
    assert fused[0].fused_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0].fused_rank == 1


def test_rrf_tie_breaker_is_stable():
    fused = reciprocal_rank_fusion(
        [RetrievalResult("b", "doc b", {"source": "b.md"})],
        [RetrievalResult("a", "doc a", {"source": "a.md"})],
        final_top_k=2,
        rrf_k=60,
    )

    assert [item.chunk_id for item in fused] == ["b", "a"]


class FakeVectorStore:
    def __init__(self, docs_and_scores):
        self.docs_and_scores = docs_and_scores
        self.last_filter = None

    def similarity_search_with_score(self, query, **kwargs):
        self.last_filter = kwargs.get("filter")
        docs = self.docs_and_scores
        metadata_filter = kwargs.get("filter")
        if metadata_filter:
            docs = [
                item
                for item in docs
                if metadata_matches_filter(item[0].metadata, metadata_filter)
            ]
        return docs[: kwargs["k"]]


def test_hybrid_retriever_modes_and_filter_application():
    docs = split_documents_with_chunk_ids(
        [
            make_doc("Nginx 502 upstream error", "data/raw/nginx.md", system="nginx"),
            make_doc("Redis OOM memory", "data/raw/redis.md", system="redis"),
        ]
    )
    vectorstore = FakeVectorStore([(docs[0], 0.1), (docs[1], 0.2)])
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_documents=docs,
        dense_top_k=2,
        bm25_top_k=2,
        rrf_k=60,
    )

    dense = retriever.retrieve("502", mode="dense", top_k=1)
    bm25 = retriever.retrieve("502", mode="bm25", top_k=1)
    hybrid = retriever.retrieve("502", mode="hybrid", top_k=2)
    filtered = retriever.retrieve(
        "502",
        mode="hybrid",
        top_k=2,
        metadata_filter={"system": "redis"},
    )

    assert dense[0].dense_rank == 1
    assert bm25[0].bm25_rank == 1
    assert hybrid[0].fused_rank == 1
    assert all(item.metadata["system"] == "redis" for item in filtered)
    assert vectorstore.last_filter == {"system": "redis"}


def test_v8_report_schema_and_query_comparison(tmp_path: Path):
    dense_results = [
        {
            "id": "q1",
            "question": "502",
            "expected_source": "nginx.md",
            "expected_system": "nginx",
            "expected_category": "application",
            "difficulty": "easy",
            "query_type": "error_code",
            "first_relevant_rank": 2,
            "hit": True,
            "latency_ms": 1.0,
            "top_k_sources": [{"source": "other.md"}, {"source": "nginx.md"}],
        }
    ]
    hybrid_results = [
        {
            **dense_results[0],
            "first_relevant_rank": 1,
            "latency_ms": 2.0,
            "top_k_sources": [{"source": "nginx.md"}],
        }
    ]
    report = build_v8_report(
        dense_results=dense_results,
        bm25_results=hybrid_results,
        hybrid_results=hybrid_results,
        document_count=2,
        chunk_count=2,
        dense_top_k=10,
        bm25_top_k=10,
        final_top_k=5,
        rrf_k=60,
        evaluated_at="2026-01-01T00:00:00",
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["dataset"]["questions"] == 1
    assert loaded["modes"]["dense"]["metrics"]["recall_at_1"] == 0.0
    assert loaded["modes"]["hybrid"]["metrics"]["recall_at_1"] == 1.0
    assert loaded["comparisons"]["top1_improvements"][0]["id"] == "q1"
