import json

import pytest
from langchain_core.documents import Document

from src.evaluate_reranker import build_report, candidate_recall, compare_modes
from src.retrieval.common import RetrievalResult, split_documents_with_chunk_ids
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranked_retriever import RerankedRetriever
from src.retrieval.reranker import CrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None
        self.batch_size = None

    def predict(self, pairs, batch_size=8):
        self.pairs = pairs
        self.batch_size = batch_size
        return self.scores[: len(pairs)]


class FakeVectorStore:
    def __init__(self, docs_and_scores):
        self.docs_and_scores = docs_and_scores
        self.last_filter = None

    def similarity_search_with_score(self, query, **kwargs):
        self.last_filter = kwargs.get("filter")
        docs = self.docs_and_scores
        if self.last_filter:
            from src.retrieval.common import metadata_matches_filter

            docs = [
                item
                for item in docs
                if metadata_matches_filter(item[0].metadata, self.last_filter)
            ]
        return docs[: kwargs["k"]]


def make_result(chunk_id, text, source, candidate_rank=None):
    return RetrievalResult(
        chunk_id=chunk_id,
        page_content=text,
        metadata={"source": source, "chunk_id": chunk_id, "system": "nginx"},
        candidate_rank=candidate_rank,
        dense_rank=candidate_rank,
        dense_score=0.1,
    )


def make_doc(text, source, **metadata):
    values = {"source": source}
    values.update(metadata)
    return Document(page_content=text, metadata=values)


def test_reranker_builds_pairs_and_scores_in_batch():
    fake = FakeCrossEncoder([0.2, 0.9])
    reranker = CrossEncoderReranker(model=fake, batch_size=2)
    candidates = [
        make_result("a", "first chunk", "a.md"),
        make_result("b", "second chunk", "b.md"),
    ]

    pairs = reranker.build_pairs("query", candidates)
    scores = reranker.score_pairs(pairs)

    assert pairs == [("query", "first chunk"), ("query", "second chunk")]
    assert scores == [0.2, 0.9]
    assert fake.batch_size == 2


def test_reranker_sorts_descending_and_preserves_metadata():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.9]))
    candidates = [
        make_result("a", "first chunk", "a.md", candidate_rank=1),
        make_result("b", "second chunk", "b.md", candidate_rank=2),
    ]

    results = reranker.rerank(
        "query",
        candidates,
        final_top_k=2,
        retrieval_mode="dense_rerank",
    )

    assert [item.chunk_id for item in results] == ["b", "a"]
    assert [item.rerank_rank for item in results] == [1, 2]
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[0].metadata["source"] == "b.md"
    assert results[0].metadata["chunk_id"] == "b"
    assert results[0].retrieval_mode == "dense_rerank"


def test_reranker_tie_breaker_uses_candidate_rank_then_chunk_id():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.5, 0.5, 0.5]))
    candidates = [
        make_result("c", "third", "c.md", candidate_rank=3),
        make_result("a", "first", "a.md", candidate_rank=1),
        make_result("b", "second", "b.md", candidate_rank=2),
    ]

    results = reranker.rerank("query", candidates, retrieval_mode="dense_rerank")

    assert [item.chunk_id for item in results] == ["a", "b", "c"]


def test_reranker_empty_cases():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([]))

    assert reranker.rerank("query", [], retrieval_mode="dense_rerank") == []
    with pytest.raises(ValueError, match="empty query"):
        reranker.rerank("", [make_result("a", "text", "a.md")], retrieval_mode="dense_rerank")


def test_reranked_retriever_dense_and_hybrid_modes_keep_candidate_trace():
    docs = split_documents_with_chunk_ids(
        [
            make_doc("Nginx 502 upstream error", "data/raw/nginx.md", system="nginx"),
            make_doc("Redis OOM memory", "data/raw/redis.md", system="redis"),
        ]
    )
    vectorstore = FakeVectorStore([(docs[0], 0.1), (docs[1], 0.2)])
    base = HybridRetriever(vectorstore=vectorstore, bm25_documents=docs)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.1, 0.9]))
    reranked = RerankedRetriever(
        base_retriever=base,
        reranker=reranker,
        candidate_top_k=2,
        final_top_k=1,
    )

    dense_results, dense_trace = reranked.retrieve("502", mode="dense_rerank")
    hybrid_results, hybrid_trace = reranked.retrieve(
        "502",
        mode="hybrid_rerank",
        metadata_filter={"system": "nginx"},
    )

    assert len(dense_trace.candidates) == 2
    assert dense_results[0].rerank_rank == 1
    assert dense_results[0].candidate_rank in {1, 2}
    assert hybrid_trace.candidate_mode == "hybrid"
    assert all(item.metadata["system"] == "nginx" for item in hybrid_results)
    assert vectorstore.last_filter == {"system": "nginx"}


def result_row(query_id, rank, source="target.md", latency=1.0, candidate=True):
    top_sources = (
        [{"source": source, "rerank_score": 0.8}]
        if rank == 1
        else [{"source": "other.md"}, {"source": source, "rerank_score": 0.4}]
    )
    return {
        "id": query_id,
        "question": "question",
        "expected_source": source,
        "expected_system": "system",
        "expected_category": "category",
        "difficulty": "easy",
        "query_type": "keyword",
        "first_relevant_rank": rank,
        "hit": rank is not None,
        "candidate_contains_expected": candidate,
        "candidate_count": 2,
        "expected_rerank_score": 0.8 if rank == 1 else 0.4,
        "latency_ms": latency,
        "candidate_latency_ms": 0.5,
        "rerank_latency_ms": 0.5,
        "top_k_sources": top_sources,
        "candidate_sources": [source] if candidate else ["other.md"],
        "candidate_top_k_sources": [{"source": source}] if candidate else [{"source": "other.md"}],
    }


def test_candidate_recall_and_mode_comparison():
    dense = [result_row("q1", 2), result_row("q2", 1)]
    hybrid = [result_row("q1", 2), result_row("q2", 1)]
    dense_rerank = [result_row("q1", 1), result_row("q2", 2, candidate=False)]
    hybrid_rerank = [result_row("q1", 1), result_row("q2", 1)]

    comparisons = compare_modes(dense, hybrid, dense_rerank, hybrid_rerank)

    assert candidate_recall(dense_rerank) == pytest.approx(0.5)
    assert comparisons["dense_top1_improvements"][0]["id"] == "q1"
    assert comparisons["dense_top1_regressions"][0]["id"] == "q2"
    assert comparisons["dense_candidate_miss"][0]["id"] == "q2"


def test_v9_report_schema_json_serializable():
    dense = [result_row("q1", 2), result_row("q2", 1, latency=2.0)]
    hybrid = [result_row("q1", 2), result_row("q2", 1, latency=2.0)]
    dense_rerank = [result_row("q1", 1, latency=3.0), result_row("q2", 1, latency=4.0)]
    hybrid_rerank = [result_row("q1", 1, latency=5.0), result_row("q2", 1, latency=6.0)]

    report = build_report(
        dense_results=dense,
        hybrid_results=hybrid,
        dense_rerank_results=dense_rerank,
        hybrid_rerank_results=hybrid_rerank,
        document_count=2,
        chunk_count=2,
        model_name="fake",
        device="cpu",
        model_load_time_ms=10.0,
        candidate_top_k=5,
        final_top_k=5,
        batch_size=2,
        evaluated_at="2026-01-01T00:00:00",
    )

    encoded = json.dumps(report)
    assert "dense_rerank" in encoded
    assert report["candidate_recall"]["dense_rerank_candidate_recall_at_5"] == 1.0
    assert report["modes"]["dense_rerank"]["latency_breakdown"]["model_load_time_ms"] == 10.0
