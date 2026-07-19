from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
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
    from ingest import COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL, load_documents
    from retrieval.common import normalize_source_name, split_documents_with_chunk_ids
    from retrieval.hybrid_retriever import HybridRetriever
except ModuleNotFoundError:
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
    from src.ingest import COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL, load_documents
    from src.retrieval.common import normalize_source_name, split_documents_with_chunk_ids
    from src.retrieval.hybrid_retriever import HybridRetriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
DEFAULT_DENSE_TOP_K = 10
DEFAULT_BM25_TOP_K = 10
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_K = 60
FOCUS_QUERY_IDS = {"disk-002", "disk-003", "nginx-005", "login-005"}


def build_vectorstore() -> Chroma:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Chroma database not found: {CHROMA_DIR}. Run src/ingest.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if get_chroma_chunk_count(vectorstore) == 0:
        raise ValueError("Chroma database is empty. Run src/ingest.py first.")

    return vectorstore


def source_rows(results) -> list[dict[str, Any]]:
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "chunk_id": result.chunk_id,
                "source": normalize_source_name(result.metadata.get("source")),
                "system": result.metadata.get("system", ""),
                "category": result.metadata.get("category", ""),
                "severity": result.metadata.get("severity", ""),
                "doc_type": result.metadata.get("doc_type", ""),
                "dense_rank": result.dense_rank,
                "dense_score": result.dense_score,
                "bm25_rank": result.bm25_rank,
                "bm25_score": result.bm25_score,
                "fused_rank": result.fused_rank,
                "fused_score": result.fused_score,
            }
        )
    return rows


def evaluate_mode(
    retriever: HybridRetriever,
    questions: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int,
) -> list[dict[str, Any]]:
    results = []
    for question in questions:
        start = time.perf_counter()
        retrieved = retriever.retrieve(
            question["question"],
            mode=mode,
            top_k=top_k,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        top_sources = source_rows(retrieved)
        first_rank = find_first_relevant_rank(
            top_sources,
            question["expected_source"],
        )
        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_source": normalize_source_name(question["expected_source"]),
                "expected_system": question["expected_system"],
                "expected_category": question["expected_category"],
                "difficulty": question["difficulty"],
                "query_type": question["query_type"],
                "first_relevant_rank": first_rank,
                "hit": first_rank is not None,
                "latency_ms": latency_ms,
                "top_k_sources": top_sources,
            }
        )
    return results


def mode_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "metrics": {
            **calculate_metrics(results),
            **calculate_latency_metrics([float(result["latency_ms"]) for result in results]),
        },
        "results": results,
    }


def metric_delta(candidate: dict[str, float], dense: dict[str, float]) -> dict[str, float]:
    keys = [
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "mrr",
        "zero_hit_rate",
        "average_retrieval_latency_ms",
        "p95_retrieval_latency_ms",
    ]
    return {key: candidate[key] - dense[key] for key in keys}


def result_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(result["id"]): result for result in results}


def compare_query_results(
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
) -> dict[str, Any]:
    dense_by_id = result_index(dense_results)
    bm25_by_id = result_index(bm25_results)
    hybrid_by_id = result_index(hybrid_results)
    per_query = []
    improvements = []
    regressions = []
    unchanged = []
    dense_only_hits = []
    bm25_only_hits = []
    hybrid_recovered = []

    for query_id, dense in dense_by_id.items():
        bm25 = bm25_by_id[query_id]
        hybrid = hybrid_by_id[query_id]
        dense_rank = dense["first_relevant_rank"]
        bm25_rank = bm25["first_relevant_rank"]
        hybrid_rank = hybrid["first_relevant_rank"]
        comparison = {
            "id": query_id,
            "question": dense["question"],
            "expected_source": dense["expected_source"],
            "difficulty": dense["difficulty"],
            "query_type": dense["query_type"],
            "dense_rank": dense_rank,
            "bm25_rank": bm25_rank,
            "hybrid_rank": hybrid_rank,
            "dense_top1": dense["top_k_sources"][0]["source"] if dense["top_k_sources"] else None,
            "bm25_top1": bm25["top_k_sources"][0]["source"] if bm25["top_k_sources"] else None,
            "hybrid_top1": hybrid["top_k_sources"][0]["source"] if hybrid["top_k_sources"] else None,
        }
        per_query.append(comparison)

        dense_top1_hit = dense_rank == 1
        bm25_top1_hit = bm25_rank == 1
        hybrid_top1_hit = hybrid_rank == 1

        if hybrid_top1_hit and not dense_top1_hit:
            improvements.append(comparison)
        elif dense_top1_hit and not hybrid_top1_hit:
            regressions.append(comparison)
        else:
            unchanged.append(comparison)

        if dense["hit"] and not bm25["hit"]:
            dense_only_hits.append(comparison)
        if bm25["hit"] and not dense["hit"]:
            bm25_only_hits.append(comparison)
        if hybrid["hit"] and not dense["hit"]:
            hybrid_recovered.append(comparison)

    return {
        "per_query": per_query,
        "top1_improvements": improvements,
        "top1_regressions": regressions,
        "unchanged_queries": unchanged,
        "dense_only_hits": dense_only_hits,
        "bm25_only_hits": bm25_only_hits,
        "hybrid_recovered_queries": hybrid_recovered,
        "focus_queries": [
            item for item in per_query if item["id"] in FOCUS_QUERY_IDS
        ],
    }


def build_v8_report(
    *,
    dense_results: list[dict[str, Any]],
    bm25_results: list[dict[str, Any]],
    hybrid_results: list[dict[str, Any]],
    document_count: int,
    chunk_count: int,
    dense_top_k: int,
    bm25_top_k: int,
    final_top_k: int,
    rrf_k: int,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    dense_summary = mode_summary(dense_results)
    bm25_summary = mode_summary(bm25_results)
    hybrid_summary = mode_summary(hybrid_results)
    comparisons = compare_query_results(dense_results, bm25_results, hybrid_results)

    return {
        "evaluated_at": evaluated_at or datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "documents": document_count,
            "chunks": chunk_count,
            "questions": len(dense_results),
        },
        "configuration": {
            "dense_top_k": dense_top_k,
            "bm25_top_k": bm25_top_k,
            "final_top_k": final_top_k,
            "rrf_k": rrf_k,
            "tokenizer": "jieba for Chinese terms plus regex extraction for IT tokens, commands, paths, numbers, IPs, and dotted/underscore/hyphen words.",
        },
        "modes": {
            "dense": dense_summary,
            "bm25": bm25_summary,
            "hybrid": hybrid_summary,
        },
        "delta_vs_dense": {
            "bm25": metric_delta(bm25_summary["metrics"], dense_summary["metrics"]),
            "hybrid": metric_delta(hybrid_summary["metrics"], dense_summary["metrics"]),
        },
        "comparisons": comparisons,
        "recommendation": recommend_default_mode(
            dense_summary["metrics"],
            hybrid_summary["metrics"],
            comparisons["top1_regressions"],
        ),
    }


def recommend_default_mode(
    dense_metrics: dict[str, float],
    hybrid_metrics: dict[str, float],
    hybrid_regressions: list[dict[str, Any]],
) -> str:
    if hybrid_regressions:
        return "Default to Dense and keep Hybrid RRF optional because Hybrid introduced Top-1 regressions."
    if hybrid_metrics["recall_at_1"] > dense_metrics["recall_at_1"] or hybrid_metrics["mrr"] > dense_metrics["mrr"]:
        return "Default to Hybrid RRF because it improves Recall@1 or MRR without Top-1 regressions in this evaluation."
    if hybrid_metrics["average_retrieval_latency_ms"] > dense_metrics["average_retrieval_latency_ms"]:
        return "Default to Dense and keep Hybrid RRF optional because quality is not better while latency is higher."
    return "Dense and Hybrid are comparable on this dataset; keep Dense as the conservative default."


def render_metrics_row(name: str, metrics: dict[str, float]) -> str:
    return (
        f"| {name} | {format_percentage(metrics['recall_at_1'])} | "
        f"{format_percentage(metrics['recall_at_3'])} | "
        f"{format_percentage(metrics['recall_at_5'])} | "
        f"{metrics['mrr']:.4f} | "
        f"{format_percentage(metrics['zero_hit_rate'])} | "
        f"{format_ms(metrics['average_retrieval_latency_ms'])} | "
        f"{format_ms(metrics['p95_retrieval_latency_ms'])} |"
    )


def render_delta_row(name: str, delta: dict[str, float]) -> str:
    return (
        f"| {name} | {delta['recall_at_1']:+.4f} | "
        f"{delta['recall_at_3']:+.4f} | "
        f"{delta['recall_at_5']:+.4f} | "
        f"{delta['mrr']:+.4f} | "
        f"{delta['zero_hit_rate']:+.4f} | "
        f"{format_ms(delta['average_retrieval_latency_ms'])} | "
        f"{format_ms(delta['p95_retrieval_latency_ms'])} |"
    )


def render_query_table(items: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Query | Expected Source | Dense Rank | BM25 Rank | Hybrid Rank | Dense Top-1 | BM25 Top-1 | Hybrid Top-1 |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    if not items:
        lines.append("| None | - | - | - | - | - | - | - |")
        return lines

    for item in items:
        lines.append(
            f"| {item['id']} | {item['expected_source']} | "
            f"{item['dense_rank'] or '-'} | {item['bm25_rank'] or '-'} | {item['hybrid_rank'] or '-'} | "
            f"{item['dense_top1'] or '-'} | {item['bm25_top1'] or '-'} | {item['hybrid_top1'] or '-'} |"
        )
    return lines


def render_markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    config = report["configuration"]
    modes = report["modes"]
    comparisons = report["comparisons"]
    lines = [
        "# V8 Hybrid Retrieval Report",
        "",
        f"- Evaluated at: {report['evaluated_at']}",
        "",
        "## Dataset",
        "",
        f"- Documents: {dataset['documents']}",
        f"- Chunks: {dataset['chunks']}",
        f"- Questions: {dataset['questions']}",
        "",
        "## Configuration",
        "",
        f"- Dense top-k: {config['dense_top_k']}",
        f"- BM25 top-k: {config['bm25_top_k']}",
        f"- Final top-k: {config['final_top_k']}",
        f"- rrf_k: {config['rrf_k']}",
        f"- Tokenizer: {config['tokenizer']}",
        "",
        "## Metrics Comparison",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        render_metrics_row("Dense", modes["dense"]["metrics"]),
        render_metrics_row("BM25", modes["bm25"]["metrics"]),
        render_metrics_row("Hybrid RRF", modes["hybrid"]["metrics"]),
        "",
        "## Delta vs V7",
        "",
        "Dense in this report uses the same expected-source matching rule as V7. Delta rows are measured against the Dense result from this V8 run.",
        "",
        "| Mode | Recall@1 Delta | Recall@3 Delta | Recall@5 Delta | MRR Delta | Zero-hit Delta | Avg Latency Delta | P95 Latency Delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        render_delta_row("BM25 - Dense", report["delta_vs_dense"]["bm25"]),
        render_delta_row("Hybrid - Dense", report["delta_vs_dense"]["hybrid"]),
        "",
        "## Non-Top-1 Query Analysis",
        "",
        *render_query_table(comparisons["focus_queries"]),
        "",
        "The four focus queries are the V7 non-Top-1 cases. They show whether lexical matching and RRF changed the ranking of known dense weaknesses.",
        "",
        "## Improvements and Regressions",
        "",
        "### Top-1 Improvements",
        "",
        *render_query_table(comparisons["top1_improvements"]),
        "",
        "### Top-1 Regressions",
        "",
        *render_query_table(comparisons["top1_regressions"]),
        "",
        "## Complementarity Analysis",
        "",
        f"- Dense-only hits: {len(comparisons['dense_only_hits'])}",
        f"- BM25-only hits: {len(comparisons['bm25_only_hits'])}",
        f"- Hybrid recovered queries: {len(comparisons['hybrid_recovered_queries'])}",
        f"- Unchanged Top-1 status queries: {len(comparisons['unchanged_queries'])}",
        "",
        "BM25 is expected to help most when a query contains exact commands, error codes, file names, service names, or configuration keys. Dense retrieval remains useful for short symptom descriptions where wording differs from the document.",
        "",
        "## Recommendation",
        "",
        report["recommendation"],
        "",
        "## Limitations",
        "",
        "- Current dataset has only 10 documents, 16 chunks, and 30 evaluation questions.",
        "- Evaluation questions come from a small self-built IT operations corpus.",
        "- Results do not represent a large enterprise knowledge base.",
        "- Recall@3 and Recall@5 were already saturated in V7.",
        "- The main decision signals for V8 are Recall@1, MRR, regressions, and latency.",
        "",
    ]
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "v8_hybrid_report.json"
    md_path = reports_dir / "v8_hybrid_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dense, BM25, and Hybrid RRF retrieval.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--dense-top-k", type=int, default=DEFAULT_DENSE_TOP_K)
    parser.add_argument("--bm25-top-k", type=int, default=DEFAULT_BM25_TOP_K)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    args = parser.parse_args()

    if min(args.dense_top_k, args.bm25_top_k, args.final_top_k, args.rrf_k) < 1:
        raise ValueError("top-k and rrf-k values must be >= 1.")

    questions = load_eval_questions(args.questions)
    raw_documents = load_documents()
    if not raw_documents:
        raise ValueError("No documents loaded from data/raw. Run src/ingest.py first.")

    validate_expected_sources(questions, raw_documents)
    vectorstore = build_vectorstore()
    chunk_count = get_chroma_chunk_count(vectorstore)
    bm25_documents = split_documents_with_chunk_ids(raw_documents)
    retriever = HybridRetriever(
        vectorstore=vectorstore,
        bm25_documents=bm25_documents,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        rrf_k=args.rrf_k,
    )

    dense_results = evaluate_mode(
        retriever,
        questions,
        mode="dense",
        top_k=args.final_top_k,
    )
    bm25_results = evaluate_mode(
        retriever,
        questions,
        mode="bm25",
        top_k=args.final_top_k,
    )
    hybrid_results = evaluate_mode(
        retriever,
        questions,
        mode="hybrid",
        top_k=args.final_top_k,
    )
    report = build_v8_report(
        dense_results=dense_results,
        bm25_results=bm25_results,
        hybrid_results=hybrid_results,
        document_count=len(raw_documents),
        chunk_count=chunk_count,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        final_top_k=args.final_top_k,
        rrf_k=args.rrf_k,
    )
    md_path, json_path = write_reports(report, args.reports_dir)

    print("Hybrid retrieval evaluation completed.")
    print(f"Questions: {report['dataset']['questions']}")
    print(f"Documents: {report['dataset']['documents']}")
    print(f"Chunks: {report['dataset']['chunks']}")
    for mode_name in ["dense", "bm25", "hybrid"]:
        metrics = report["modes"][mode_name]["metrics"]
        print(f"{mode_name.title()} Recall@1: {format_percentage(metrics['recall_at_1'])}")
        print(f"{mode_name.title()} Recall@3: {format_percentage(metrics['recall_at_3'])}")
        print(f"{mode_name.title()} Recall@5: {format_percentage(metrics['recall_at_5'])}")
        print(f"{mode_name.title()} MRR: {metrics['mrr']:.4f}")
        print(f"{mode_name.title()} Zero-hit Rate: {format_percentage(metrics['zero_hit_rate'])}")
        print(f"{mode_name.title()} Average Retrieval Latency: {format_ms(metrics['average_retrieval_latency_ms'])}")
        print(f"{mode_name.title()} P95 Retrieval Latency: {format_ms(metrics['p95_retrieval_latency_ms'])}")
    print(f"Top-1 improvements: {len(report['comparisons']['top1_improvements'])}")
    print(f"Top-1 regressions: {len(report['comparisons']['top1_regressions'])}")
    print(f"Recommendation: {report['recommendation']}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
