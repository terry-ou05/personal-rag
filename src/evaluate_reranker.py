from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from evaluate_hybrid_retrieval import (
        DEFAULT_BM25_TOP_K,
        DEFAULT_DENSE_TOP_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_QUESTIONS_PATH,
        DEFAULT_REPORTS_DIR,
        DEFAULT_RRF_K,
        build_vectorstore,
        evaluate_mode,
        mode_summary,
        source_rows,
    )
    from evaluate_retrieval import (
        calculate_latency_metrics,
        calculate_metrics,
        find_first_relevant_rank,
        format_ms,
        format_percentage,
        get_chroma_chunk_count,
        load_eval_questions,
        validate_expected_sources,
    )
    from ingest import load_documents
    from retrieval.common import normalize_source_name, split_documents_with_chunk_ids
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranked_retriever import RerankedRetriever
    from retrieval.reranker import (
        DEFAULT_RERANKER_BATCH_SIZE,
        DEFAULT_RERANKER_DEVICE,
        DEFAULT_RERANKER_MODEL,
        CrossEncoderReranker,
    )
except ModuleNotFoundError:
    from src.evaluate_hybrid_retrieval import (
        DEFAULT_BM25_TOP_K,
        DEFAULT_DENSE_TOP_K,
        DEFAULT_FINAL_TOP_K,
        DEFAULT_QUESTIONS_PATH,
        DEFAULT_REPORTS_DIR,
        DEFAULT_RRF_K,
        build_vectorstore,
        evaluate_mode,
        mode_summary,
        source_rows,
    )
    from src.evaluate_retrieval import (
        calculate_latency_metrics,
        calculate_metrics,
        find_first_relevant_rank,
        format_ms,
        format_percentage,
        get_chroma_chunk_count,
        load_eval_questions,
        validate_expected_sources,
    )
    from src.ingest import load_documents
    from src.retrieval.common import normalize_source_name, split_documents_with_chunk_ids
    from src.retrieval.hybrid_retriever import HybridRetriever
    from src.retrieval.reranked_retriever import RerankedRetriever
    from src.retrieval.reranker import (
        DEFAULT_RERANKER_BATCH_SIZE,
        DEFAULT_RERANKER_DEVICE,
        DEFAULT_RERANKER_MODEL,
        CrossEncoderReranker,
    )


FOCUS_QUERY_IDS = {"disk-002", "disk-003", "nginx-005", "login-005", "login-003"}


def evaluate_reranked_mode(
    reranked_retriever: RerankedRetriever,
    questions: list[dict[str, Any]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    results = []
    for question in questions:
        retrieved, trace = reranked_retriever.retrieve(
            question["question"],
            mode=mode,
        )
        top_sources = source_rows(retrieved)
        candidate_source_rows = source_rows(trace.candidates)
        first_rank = find_first_relevant_rank(
            top_sources,
            question["expected_source"],
        )
        expected_source = normalize_source_name(question["expected_source"])
        expected_source_row = next(
            (source for source in top_sources if source["source"] == expected_source),
            None,
        )
        candidate_sources = [
            source["source"]
            for source in candidate_source_rows
        ]
        candidate_contains_expected = any(
            source["source"] == expected_source for source in candidate_source_rows
        )
        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_source": expected_source,
                "expected_system": question["expected_system"],
                "expected_category": question["expected_category"],
                "difficulty": question["difficulty"],
                "query_type": question["query_type"],
                "first_relevant_rank": first_rank,
                "hit": first_rank is not None,
                "candidate_contains_expected": candidate_contains_expected,
                "candidate_count": trace.candidate_count,
                "expected_rerank_score": (
                    expected_source_row.get("rerank_score") if expected_source_row else None
                ),
                "latency_ms": trace.end_to_end_latency_ms,
                "candidate_latency_ms": trace.candidate_latency_ms,
                "rerank_latency_ms": trace.rerank_latency_ms,
                "top_k_sources": top_sources,
                "candidate_sources": candidate_sources,
                "candidate_top_k_sources": candidate_source_rows,
            }
        )
    return results


def candidate_recall(results: list[dict[str, Any]]) -> float:
    if not results:
        raise ValueError("Cannot calculate candidate recall for zero results.")
    return sum(1 for result in results if result["candidate_contains_expected"]) / len(results)


def reranked_summary(results: list[dict[str, Any]], *, load_time_ms: float) -> dict[str, Any]:
    summary = mode_summary(results)
    rerank_latencies = [float(result["rerank_latency_ms"]) for result in results]
    end_to_end_latencies = [float(result["latency_ms"]) for result in results]
    warm_rerank_latencies = rerank_latencies[1:] or rerank_latencies
    warm_end_to_end = end_to_end_latencies[1:] or end_to_end_latencies
    summary["candidate_recall_at_5"] = candidate_recall(results)
    summary["latency_breakdown"] = {
        "model_load_time_ms": load_time_ms,
        "cold_first_query_latency_ms": end_to_end_latencies[0],
        "warm_average_rerank_latency_ms": calculate_latency_metrics(warm_rerank_latencies)[
            "average_retrieval_latency_ms"
        ],
        "warm_p95_rerank_latency_ms": calculate_latency_metrics(warm_rerank_latencies)[
            "p95_retrieval_latency_ms"
        ],
        "end_to_end_average_latency_ms": calculate_latency_metrics(end_to_end_latencies)[
            "average_retrieval_latency_ms"
        ],
        "end_to_end_p95_latency_ms": calculate_latency_metrics(end_to_end_latencies)[
            "p95_retrieval_latency_ms"
        ],
        "warm_average_end_to_end_latency_ms": calculate_latency_metrics(warm_end_to_end)[
            "average_retrieval_latency_ms"
        ],
        "warm_p95_end_to_end_latency_ms": calculate_latency_metrics(warm_end_to_end)[
            "p95_retrieval_latency_ms"
        ],
        "average_candidates_reranked_per_query": sum(
            result["candidate_count"] for result in results
        )
        / len(results),
    }
    return summary


def metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    keys = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "zero_hit_rate",
        "average_retrieval_latency_ms",
        "p95_retrieval_latency_ms",
    ]
    return {key: candidate[key] - baseline[key] for key in keys}


def by_id(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {result["id"]: result for result in results}


def rank_status(base_rank: int | None, candidate_rank_value: int | None) -> str:
    if base_rank is None and candidate_rank_value is None:
        return "candidate_miss"
    if base_rank is None:
        return "recovered"
    if candidate_rank_value is None:
        return "candidate_miss"
    if candidate_rank_value < base_rank:
        return "rank_improved"
    if candidate_rank_value > base_rank:
        return "rank_worsened"
    return "rank_unchanged"


def compare_modes(
    dense_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    dense_rerank_results: list[dict[str, Any]],
    hybrid_rerank_results: list[dict[str, Any]],
) -> dict[str, Any]:
    dense = by_id(dense_results)
    hybrid = by_id(hybrid_results)
    dense_rerank = by_id(dense_rerank_results)
    hybrid_rerank = by_id(hybrid_rerank_results)
    per_query = []
    dense_top1_improvements = []
    dense_top1_regressions = []
    hybrid_top1_improvements = []
    hybrid_top1_regressions = []

    for query_id, dense_item in dense.items():
        hybrid_item = hybrid[query_id]
        dense_rerank_item = dense_rerank[query_id]
        hybrid_rerank_item = hybrid_rerank[query_id]
        comparison = {
            "id": query_id,
            "question": dense_item["question"],
            "expected_source": dense_item["expected_source"],
            "dense_rank": dense_item["first_relevant_rank"],
            "hybrid_rank": hybrid_item["first_relevant_rank"],
            "dense_rerank_rank": dense_rerank_item["first_relevant_rank"],
            "hybrid_rerank_rank": hybrid_rerank_item["first_relevant_rank"],
            "dense_top1": dense_item["top_k_sources"][0]["source"] if dense_item["top_k_sources"] else None,
            "hybrid_top1": hybrid_item["top_k_sources"][0]["source"] if hybrid_item["top_k_sources"] else None,
            "dense_rerank_top1": dense_rerank_item["top_k_sources"][0]["source"] if dense_rerank_item["top_k_sources"] else None,
            "hybrid_rerank_top1": hybrid_rerank_item["top_k_sources"][0]["source"] if hybrid_rerank_item["top_k_sources"] else None,
            "dense_candidate_contains_expected": dense_rerank_item["candidate_contains_expected"],
            "hybrid_candidate_contains_expected": hybrid_rerank_item["candidate_contains_expected"],
            "dense_expected_rerank_score": dense_rerank_item["expected_rerank_score"],
            "hybrid_expected_rerank_score": hybrid_rerank_item["expected_rerank_score"],
            "dense_rerank_status": rank_status(
                dense_item["first_relevant_rank"],
                dense_rerank_item["first_relevant_rank"],
            ),
            "hybrid_rerank_status_vs_hybrid": rank_status(
                hybrid_item["first_relevant_rank"],
                hybrid_rerank_item["first_relevant_rank"],
            ),
            "hybrid_rerank_status_vs_dense": rank_status(
                dense_item["first_relevant_rank"],
                hybrid_rerank_item["first_relevant_rank"],
            ),
            "dense_latency_ms": dense_item["latency_ms"],
            "hybrid_latency_ms": hybrid_item["latency_ms"],
            "dense_rerank_latency_ms": dense_rerank_item["latency_ms"],
            "hybrid_rerank_latency_ms": hybrid_rerank_item["latency_ms"],
        }
        per_query.append(comparison)

        if dense_item["first_relevant_rank"] != 1 and dense_rerank_item["first_relevant_rank"] == 1:
            dense_top1_improvements.append(comparison)
        if dense_item["first_relevant_rank"] == 1 and dense_rerank_item["first_relevant_rank"] != 1:
            dense_top1_regressions.append(comparison)
        if hybrid_item["first_relevant_rank"] != 1 and hybrid_rerank_item["first_relevant_rank"] == 1:
            hybrid_top1_improvements.append(comparison)
        if hybrid_item["first_relevant_rank"] == 1 and hybrid_rerank_item["first_relevant_rank"] != 1:
            hybrid_top1_regressions.append(comparison)

    return {
        "per_query": per_query,
        "focus_queries": [item for item in per_query if item["id"] in FOCUS_QUERY_IDS],
        "dense_top1_improvements": dense_top1_improvements,
        "dense_top1_regressions": dense_top1_regressions,
        "hybrid_top1_improvements": hybrid_top1_improvements,
        "hybrid_top1_regressions": hybrid_top1_regressions,
        "dense_rank_improved": [
            item for item in per_query if item["dense_rerank_status"] == "rank_improved"
        ],
        "dense_rank_worsened": [
            item for item in per_query if item["dense_rerank_status"] == "rank_worsened"
        ],
        "dense_candidate_miss": [
            item for item in per_query if not item["dense_candidate_contains_expected"]
        ],
    }


def recommend(report: dict[str, Any]) -> str:
    dense = report["modes"]["dense"]["metrics"]
    dense_rerank = report["modes"]["dense_rerank"]["metrics"]
    regressions = report["comparisons"]["dense_top1_regressions"]
    if (
        dense_rerank["recall_at_1"] > dense["recall_at_1"]
        and dense_rerank["mrr"] > dense["mrr"]
        and not regressions
    ):
        return "Default to Dense + Reranker if the observed latency is acceptable for local use."
    if regressions:
        return "Keep Dense as the default because Dense + Reranker introduced Top-1 regressions."
    return "Keep Dense as the default and keep Reranker as an experiment because ranking quality did not clearly improve."


def build_report(
    *,
    dense_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    dense_rerank_results: list[dict[str, Any]],
    hybrid_rerank_results: list[dict[str, Any]],
    document_count: int,
    chunk_count: int,
    model_name: str,
    device: str,
    model_load_time_ms: float,
    candidate_top_k: int,
    final_top_k: int,
    batch_size: int,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    dense_summary = mode_summary(dense_results)
    hybrid_summary = mode_summary(hybrid_results)
    dense_rerank_summary = reranked_summary(
        dense_rerank_results,
        load_time_ms=model_load_time_ms,
    )
    hybrid_rerank_summary = reranked_summary(
        hybrid_rerank_results,
        load_time_ms=model_load_time_ms,
    )
    comparisons = compare_modes(
        dense_results,
        hybrid_results,
        dense_rerank_results,
        hybrid_rerank_results,
    )
    report = {
        "evaluated_at": evaluated_at or datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "documents": document_count,
            "chunks": chunk_count,
            "questions": len(dense_results),
        },
        "model": {
            "name": model_name,
            "device": device,
            "candidate_top_k": candidate_top_k,
            "final_top_k": final_top_k,
            "batch_size": batch_size,
        },
        "modes": {
            "dense": dense_summary,
            "hybrid": hybrid_summary,
            "dense_rerank": dense_rerank_summary,
            "hybrid_rerank": hybrid_rerank_summary,
        },
        "candidate_recall": {
            "dense_rerank_candidate_recall_at_5": dense_rerank_summary[
                "candidate_recall_at_5"
            ],
            "hybrid_rerank_candidate_recall_at_5": hybrid_rerank_summary[
                "candidate_recall_at_5"
            ],
        },
        "metric_delta": {
            "dense_rerank_vs_dense": metric_delta(
                dense_rerank_summary["metrics"],
                dense_summary["metrics"],
            ),
            "hybrid_rerank_vs_hybrid": metric_delta(
                hybrid_rerank_summary["metrics"],
                hybrid_summary["metrics"],
            ),
            "hybrid_rerank_vs_dense": metric_delta(
                hybrid_rerank_summary["metrics"],
                dense_summary["metrics"],
            ),
        },
        "comparisons": comparisons,
    }
    report["recommendation"] = recommend(report)
    return report


def metrics_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {format_percentage(metrics['recall_at_1'])} | "
        f"{format_percentage(metrics['recall_at_3'])} | "
        f"{format_percentage(metrics['recall_at_5'])} | "
        f"{metrics['mrr']:.4f} | "
        f"{format_percentage(metrics['zero_hit_rate'])} | "
        f"{format_ms(metrics['average_retrieval_latency_ms'])} | "
        f"{format_ms(metrics['p95_retrieval_latency_ms'])} |"
    )


def delta_row(name: str, delta: dict[str, float]) -> str:
    return (
        f"| {name} | {delta['recall_at_1']:+.4f} | "
        f"{delta['recall_at_3']:+.4f} | "
        f"{delta['recall_at_5']:+.4f} | "
        f"{delta['mrr']:+.4f} | "
        f"{delta['zero_hit_rate']:+.4f} | "
        f"{format_ms(delta['average_retrieval_latency_ms'])} | "
        f"{format_ms(delta['p95_retrieval_latency_ms'])} |"
    )


def comparison_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    if not items:
        lines.append("| None | - | - | - | - | - | - | - |")
        return lines
    for item in items:
        lines.append(
            f"| {item['id']} | {item['expected_source']} | "
            f"{item['dense_rank'] or '-'} | {item['hybrid_rank'] or '-'} | "
            f"{item['dense_rerank_rank'] or '-'} | {item['hybrid_rerank_rank'] or '-'} | "
            f"{item['dense_rerank_top1'] or '-'} | {item['hybrid_rerank_top1'] or '-'} |"
        )
    return lines


def latency_lines(name: str, summary: dict[str, Any]) -> list[str]:
    item = summary["latency_breakdown"]
    return [
        f"### {name}",
        "",
        f"- Model load time: {format_ms(item['model_load_time_ms'])}",
        f"- Cold first query: {format_ms(item['cold_first_query_latency_ms'])}",
        f"- Warm average rerank latency: {format_ms(item['warm_average_rerank_latency_ms'])}",
        f"- Warm P95 rerank latency: {format_ms(item['warm_p95_rerank_latency_ms'])}",
        f"- End-to-end average latency: {format_ms(item['end_to_end_average_latency_ms'])}",
        f"- End-to-end P95 latency: {format_ms(item['end_to_end_p95_latency_ms'])}",
        f"- Average candidates reranked per query: {item['average_candidates_reranked_per_query']:.2f}",
        "",
    ]


def render_markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    model = report["model"]
    modes = report["modes"]
    comparisons = report["comparisons"]
    lines = [
        "# V9 Cross-Encoder Reranker Evaluation",
        "",
        f"- Evaluated at: {report['evaluated_at']}",
        "",
        "## Dataset",
        "",
        f"- Documents: {dataset['documents']}",
        f"- Chunks: {dataset['chunks']}",
        f"- Questions: {dataset['questions']}",
        "",
        "## Model Configuration",
        "",
        f"- Model name: {model['name']}",
        f"- Device: {model['device']}",
        f"- Candidate top-k: {model['candidate_top_k']}",
        f"- Final top-k: {model['final_top_k']}",
        f"- Batch size: {model['batch_size']}",
        "",
        "## Baseline Metrics",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metrics_row("Dense", modes["dense"]["metrics"]),
        metrics_row("Hybrid RRF", modes["hybrid"]["metrics"]),
        "",
        "## Reranker Metrics",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        metrics_row("Dense + Reranker", modes["dense_rerank"]["metrics"]),
        metrics_row("Hybrid + Reranker", modes["hybrid_rerank"]["metrics"]),
        "",
        "## Candidate Recall",
        "",
        f"- Dense + Reranker Candidate Recall@5: {format_percentage(report['candidate_recall']['dense_rerank_candidate_recall_at_5'])}",
        f"- Hybrid + Reranker Candidate Recall@5: {format_percentage(report['candidate_recall']['hybrid_rerank_candidate_recall_at_5'])}",
        "",
        "## Metric Delta",
        "",
        "| Comparison | Recall@1 Delta | Recall@3 Delta | Recall@5 Delta | MRR Delta | Zero-hit Delta | Avg Latency Delta | P95 Latency Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        delta_row("Dense+Rerank vs Dense", report["metric_delta"]["dense_rerank_vs_dense"]),
        delta_row("Hybrid+Rerank vs Hybrid", report["metric_delta"]["hybrid_rerank_vs_hybrid"]),
        delta_row("Hybrid+Rerank vs Dense", report["metric_delta"]["hybrid_rerank_vs_dense"]),
        "",
        "## Cold and Warm Latency",
        "",
        *latency_lines("Dense + Reranker", modes["dense_rerank"]),
        *latency_lines("Hybrid + Reranker", modes["hybrid_rerank"]),
        "## Non-Top-1 Analysis",
        "",
        *comparison_table(comparisons["focus_queries"]),
        "",
        "## Improvements and Regressions",
        "",
        "### Dense + Reranker Top-1 Improvements",
        "",
        *comparison_table(comparisons["dense_top1_improvements"]),
        "",
        "### Dense + Reranker Top-1 Regressions",
        "",
        *comparison_table(comparisons["dense_top1_regressions"]),
        "",
        "### Rank Improvements",
        "",
        *comparison_table(comparisons["dense_rank_improved"]),
        "",
        "### Rank Regressions",
        "",
        *comparison_table(comparisons["dense_rank_worsened"]),
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
        "## Limitations",
        "",
        "- Current dataset has only 10 documents, 16 chunks, and 30 evaluation queries.",
        "- Test questions come from a small self-built IT operations corpus.",
        "- Results do not represent a large enterprise knowledge base.",
        "- Cross-Encoder reranking adds compute cost and model load time.",
        "- This evaluates retrieval ranking only, not final answer generation quality.",
        "- Query Rewrite, retrieval sufficiency checks, Qdrant, and LangGraph are not implemented in V9.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "v9_reranker_report.json"
    md_path = reports_dir / "v9_reranker_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder reranking.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--model-name", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--device", default=DEFAULT_RERANKER_DEVICE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_RERANKER_BATCH_SIZE)
    parser.add_argument("--candidate-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    args = parser.parse_args()

    if min(args.batch_size, args.candidate_top_k, args.final_top_k) < 1:
        raise ValueError("batch-size and top-k values must be >= 1.")

    questions = load_eval_questions(args.questions)
    raw_documents = load_documents()
    if not raw_documents:
        raise ValueError("No documents loaded from data/raw. Run src/ingest.py first.")
    validate_expected_sources(questions, raw_documents)

    vectorstore = build_vectorstore()
    chunk_count = get_chroma_chunk_count(vectorstore)
    chunks = split_documents_with_chunk_ids(raw_documents)
    base_retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_documents=chunks,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
    )

    dense_results = evaluate_mode(
        base_retriever,
        questions,
        mode="dense",
        top_k=args.final_top_k,
    )
    hybrid_results = evaluate_mode(
        base_retriever,
        questions,
        mode="hybrid",
        top_k=args.final_top_k,
    )

    reranker = CrossEncoderReranker(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
    )
    reranked_retriever = RerankedRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
    )
    dense_rerank_results = evaluate_reranked_mode(
        reranked_retriever,
        questions,
        mode="dense_rerank",
    )
    hybrid_rerank_results = evaluate_reranked_mode(
        reranked_retriever,
        questions,
        mode="hybrid_rerank",
    )

    report = build_report(
        dense_results=dense_results,
        hybrid_results=hybrid_results,
        dense_rerank_results=dense_rerank_results,
        hybrid_rerank_results=hybrid_rerank_results,
        document_count=len(raw_documents),
        chunk_count=chunk_count,
        model_name=reranker.model_name,
        device=reranker.device,
        model_load_time_ms=reranker.load_time_ms,
        candidate_top_k=args.candidate_top_k,
        final_top_k=args.final_top_k,
        batch_size=args.batch_size,
    )
    md_path, json_path = write_reports(report, args.reports_dir)

    print("Reranker evaluation completed.")
    print(f"Questions: {report['dataset']['questions']}")
    print(f"Documents: {report['dataset']['documents']}")
    print(f"Chunks: {report['dataset']['chunks']}")
    print(f"Model: {report['model']['name']}")
    print(f"Device: {report['model']['device']}")
    for mode_name in ["dense", "hybrid", "dense_rerank", "hybrid_rerank"]:
        metrics = report["modes"][mode_name]["metrics"]
        print(f"{mode_name} Recall@1: {format_percentage(metrics['recall_at_1'])}")
        print(f"{mode_name} Recall@3: {format_percentage(metrics['recall_at_3'])}")
        print(f"{mode_name} Recall@5: {format_percentage(metrics['recall_at_5'])}")
        print(f"{mode_name} MRR: {metrics['mrr']:.4f}")
        print(f"{mode_name} Zero-hit Rate: {format_percentage(metrics['zero_hit_rate'])}")
    print(f"Dense+Rerank Candidate Recall@5: {format_percentage(report['candidate_recall']['dense_rerank_candidate_recall_at_5'])}")
    print(f"Hybrid+Rerank Candidate Recall@5: {format_percentage(report['candidate_recall']['hybrid_rerank_candidate_recall_at_5'])}")
    print(f"Model load time: {format_ms(report['modes']['dense_rerank']['latency_breakdown']['model_load_time_ms'])}")
    print(f"Dense+Rerank cold first query: {format_ms(report['modes']['dense_rerank']['latency_breakdown']['cold_first_query_latency_ms'])}")
    print(f"Dense+Rerank warm avg rerank latency: {format_ms(report['modes']['dense_rerank']['latency_breakdown']['warm_average_rerank_latency_ms'])}")
    print(f"Dense+Rerank warm P95 rerank latency: {format_ms(report['modes']['dense_rerank']['latency_breakdown']['warm_p95_rerank_latency_ms'])}")
    print(f"Top-1 improvements: {len(report['comparisons']['dense_top1_improvements'])}")
    print(f"Top-1 regressions: {len(report['comparisons']['dense_top1_regressions'])}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
