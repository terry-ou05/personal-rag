from __future__ import annotations

from .common import RetrievalResult


def reciprocal_rank_fusion(
    dense_results: list[RetrievalResult],
    bm25_results: list[RetrievalResult],
    *,
    final_top_k: int = 5,
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1.")
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1.")

    fused: dict[str, RetrievalResult] = {}
    scores: dict[str, float] = {}

    def add_result(result: RetrievalResult, rank: int, source: str) -> None:
        if result.chunk_id not in fused:
            fused[result.chunk_id] = RetrievalResult(
                chunk_id=result.chunk_id,
                page_content=result.page_content,
                metadata=dict(result.metadata),
                retrieval_mode="hybrid",
            )
            scores[result.chunk_id] = 0.0

        target = fused[result.chunk_id]
        scores[result.chunk_id] += 1 / (rrf_k + rank)

        if source == "dense":
            target.dense_rank = rank
            target.dense_score = result.dense_score
        elif source == "bm25":
            target.bm25_rank = rank
            target.bm25_score = result.bm25_score

    for rank, result in enumerate(dense_results, start=1):
        add_result(result, rank, "dense")

    for rank, result in enumerate(bm25_results, start=1):
        add_result(result, rank, "bm25")

    ranked = sorted(
        fused.values(),
        key=lambda result: (
            -scores[result.chunk_id],
            result.dense_rank is None,
            result.dense_rank or 10**9,
            result.bm25_rank is None,
            result.bm25_rank or 10**9,
            result.chunk_id,
        ),
    )

    output = []
    for fused_rank, result in enumerate(ranked[:final_top_k], start=1):
        result.fused_rank = fused_rank
        result.candidate_rank = fused_rank
        result.fused_score = scores[result.chunk_id]
        output.append(result)
    return output
