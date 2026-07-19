from __future__ import annotations

import time
from dataclasses import dataclass

from .common import RetrievalResult
from .hybrid_retriever import HybridRetriever
from .reranker import CrossEncoderReranker


@dataclass
class RerankTrace:
    candidate_count: int
    candidate_latency_ms: float
    rerank_latency_ms: float
    end_to_end_latency_ms: float
    candidate_mode: str
    candidates: list[RetrievalResult]


class RerankedRetriever:
    def __init__(
        self,
        *,
        base_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        candidate_top_k: int = 5,
        final_top_k: int = 5,
    ) -> None:
        if candidate_top_k < 1:
            raise ValueError("candidate_top_k must be >= 1.")
        if final_top_k < 1:
            raise ValueError("final_top_k must be >= 1.")
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.candidate_top_k = candidate_top_k
        self.final_top_k = final_top_k

    def retrieve(
        self,
        query: str,
        *,
        mode: str = "dense_rerank",
        metadata_filter: dict | None = None,
    ) -> tuple[list[RetrievalResult], RerankTrace]:
        mode = mode.lower()
        if mode not in {"dense_rerank", "hybrid_rerank"}:
            raise ValueError(f"Unsupported reranked retrieval mode: {mode}")

        candidate_mode = "dense" if mode == "dense_rerank" else "hybrid"
        start = time.perf_counter()
        candidate_start = time.perf_counter()
        candidates = self.base_retriever.retrieve(
            query,
            mode=candidate_mode,
            top_k=self.candidate_top_k,
            metadata_filter=metadata_filter,
        )
        candidate_latency_ms = (time.perf_counter() - candidate_start) * 1000

        rerank_start = time.perf_counter()
        results = self.reranker.rerank(
            query,
            candidates,
            final_top_k=self.final_top_k,
            retrieval_mode=mode,
        )
        rerank_latency_ms = (time.perf_counter() - rerank_start) * 1000
        end_to_end_latency_ms = (time.perf_counter() - start) * 1000

        return results, RerankTrace(
            candidate_count=len(candidates),
            candidate_latency_ms=candidate_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
            end_to_end_latency_ms=end_to_end_latency_ms,
            candidate_mode=candidate_mode,
            candidates=candidates,
        )
