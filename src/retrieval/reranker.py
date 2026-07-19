from __future__ import annotations

import time
from typing import Protocol

from .common import RetrievalResult, clone_result


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"
DEFAULT_RERANKER_DEVICE = "cpu"
DEFAULT_RERANKER_BATCH_SIZE = 8


class PairScoringModel(Protocol):
    def predict(self, pairs, batch_size: int = 8):
        ...


class CrossEncoderReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = DEFAULT_RERANKER_DEVICE,
        batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        model: PairScoringModel | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.load_time_ms = 0.0

        if model is not None:
            self.model = model
            return

        start = time.perf_counter()
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for Cross-Encoder reranking."
            ) from exc

        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load reranker model '{model_name}' on device '{device}'."
            ) from exc
        self.load_time_ms = (time.perf_counter() - start) * 1000

    def build_pairs(self, query: str, candidates: list[RetrievalResult]) -> list[tuple[str, str]]:
        if not query or not query.strip():
            raise ValueError("Cannot rerank an empty query.")
        return [(query, candidate.page_content) for candidate in candidates]

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        return [float(score) for score in scores]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        *,
        final_top_k: int = 5,
        retrieval_mode: str,
    ) -> list[RetrievalResult]:
        if final_top_k < 1:
            raise ValueError("final_top_k must be >= 1.")
        if not candidates:
            return []

        pairs = self.build_pairs(query, candidates)
        scores = self.score_pairs(pairs)
        scored_results = []
        for index, (candidate, score) in enumerate(zip(candidates, scores), start=1):
            item = clone_result(candidate)
            item.retrieval_mode = retrieval_mode
            item.candidate_rank = item.candidate_rank or index
            item.rerank_score = score
            scored_results.append(item)

        ranked = sorted(
            scored_results,
            key=lambda result: (
                -float(result.rerank_score or 0.0),
                result.candidate_rank or 10**9,
                result.chunk_id,
            ),
        )
        output = []
        for rank, result in enumerate(ranked[:final_top_k], start=1):
            result.rerank_rank = rank
            output.append(result)
        return output
