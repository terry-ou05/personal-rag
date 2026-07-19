from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from ingest import COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL, load_documents
except ModuleNotFoundError:
    from src.ingest import COLLECTION_NAME, CHROMA_DIR, EMBEDDING_MODEL, load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
REQUIRED_FIELDS = {
    "id",
    "question",
    "expected_source",
    "expected_system",
    "expected_category",
    "difficulty",
    "query_type",
}


def normalize_source(value: str | Path | None) -> str:
    if value is None:
        return ""
    normalized = str(value).replace("\\", "/").strip()
    return Path(normalized).name


def load_eval_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation questions file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in evaluation questions file: {path}") from exc

    if not isinstance(data, list):
        raise ValueError("Evaluation questions file must contain a JSON list.")

    questions = []
    seen_ids = set()
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question item #{index} must be a JSON object.")

        missing = REQUIRED_FIELDS - item.keys()
        if missing:
            raise ValueError(
                f"Question item #{index} is missing required fields: {sorted(missing)}"
            )

        question_id = str(item["id"]).strip()
        if not question_id:
            raise ValueError(f"Question item #{index} has an empty id.")
        if question_id in seen_ids:
            raise ValueError(f"Duplicate evaluation question id: {question_id}")
        seen_ids.add(question_id)

        questions.append(dict(item))

    return questions


def validate_expected_sources(questions: list[dict[str, Any]], raw_documents: list) -> None:
    available_sources = {
        normalize_source(document.metadata.get("source")) for document in raw_documents
    }
    missing = sorted(
        {
            normalize_source(question["expected_source"])
            for question in questions
            if normalize_source(question["expected_source"]) not in available_sources
        }
    )
    if missing:
        raise ValueError(
            "Evaluation questions reference sources that are not in the current "
            f"loaded documents: {missing}"
        )


def get_chroma_chunk_count(vectorstore: Chroma) -> int:
    try:
        return int(vectorstore._collection.count())
    except Exception as exc:
        raise RuntimeError("Unable to count Chroma chunks. Rebuild the knowledge base.") from exc


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


def retrieve_sources(vectorstore: Chroma, question: str, top_k: int = 5) -> tuple[list[dict[str, Any]], float]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    start = time.perf_counter()
    docs = retriever.invoke(question)
    latency_ms = (time.perf_counter() - start) * 1000

    sources = []
    for rank, doc in enumerate(docs, start=1):
        sources.append(
            {
                "rank": rank,
                "source": normalize_source(doc.metadata.get("source")),
                "system": doc.metadata.get("system", ""),
                "category": doc.metadata.get("category", ""),
                "severity": doc.metadata.get("severity", ""),
                "doc_type": doc.metadata.get("doc_type", ""),
            }
        )
    return sources, latency_ms


def find_first_relevant_rank(retrieved_sources: list[dict[str, Any] | str], expected_source: str) -> int | None:
    expected = normalize_source(expected_source)
    for index, item in enumerate(retrieved_sources, start=1):
        source = item.get("source") if isinstance(item, dict) else item
        if normalize_source(source) == expected:
            return index
    return None


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, float]:
    if not results:
        raise ValueError("Cannot calculate metrics for zero evaluation results.")

    total = len(results)

    def recall_at(k: int) -> float:
        hits = sum(
            1
            for result in results
            if result["first_relevant_rank"] is not None
            and result["first_relevant_rank"] <= k
        )
        return hits / total

    reciprocal_ranks = [
        0.0 if result["first_relevant_rank"] is None else 1 / result["first_relevant_rank"]
        for result in results
    ]
    zero_hits = sum(1 for result in results if result["first_relevant_rank"] is None)

    return {
        "recall_at_1": recall_at(1),
        "recall_at_3": recall_at(3),
        "recall_at_5": recall_at(5),
        "mrr": sum(reciprocal_ranks) / total,
        "zero_hit_rate": zero_hits / total,
    }


def calculate_latency_metrics(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        raise ValueError("Cannot calculate latency metrics for empty latency list.")

    sorted_latencies = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(sorted_latencies)) - 1)
    return {
        "average_retrieval_latency_ms": statistics.mean(sorted_latencies),
        "p95_retrieval_latency_ms": sorted_latencies[p95_index],
    }


def _group_results(results: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for result in results:
        grouped[str(result.get(field, ""))].append(result)

    return {
        key: {
            "count": len(items),
            **calculate_metrics(items),
        }
        for key, items in sorted(grouped.items())
    }


def evaluate_questions(
    vectorstore: Chroma,
    questions: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    results = []
    for question in questions:
        retrieved_sources, latency_ms = retrieve_sources(
            vectorstore,
            question["question"],
            top_k=top_k,
        )
        first_rank = find_first_relevant_rank(
            retrieved_sources,
            question["expected_source"],
        )

        results.append(
            {
                "id": question["id"],
                "question": question["question"],
                "expected_source": normalize_source(question["expected_source"]),
                "expected_system": question["expected_system"],
                "expected_category": question["expected_category"],
                "difficulty": question["difficulty"],
                "query_type": question["query_type"],
                "first_relevant_rank": first_rank,
                "hit": first_rank is not None,
                "latency_ms": latency_ms,
                "top_k_sources": retrieved_sources,
            }
        )
    return results


def build_evaluation_report(
    results: list[dict[str, Any]],
    document_count: int,
    chunk_count: int,
    evaluated_at: str | None = None,
) -> dict[str, Any]:
    evaluated_at = evaluated_at or datetime.now().isoformat(timespec="seconds")
    metrics = calculate_metrics(results)
    latency_metrics = calculate_latency_metrics(
        [float(result["latency_ms"]) for result in results]
    )
    failed_queries = [
        result for result in results if result["first_relevant_rank"] is None
    ]

    return {
        "evaluated_at": evaluated_at,
        "document_count": document_count,
        "chunk_count": chunk_count,
        "question_count": len(results),
        "metrics": {**metrics, **latency_metrics},
        "by_difficulty": _group_results(results, "difficulty"),
        "by_system": _group_results(results, "expected_system"),
        "by_category": _group_results(results, "expected_category"),
        "failed_queries": failed_queries,
        "results": results,
    }


def format_percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_ms(value: float) -> str:
    return f"{value:.2f} ms"


def render_markdown_report(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    lines = [
        "# V7 Retrieval Evaluation Baseline",
        "",
        f"- Evaluated at: {report['evaluated_at']}",
        f"- Document count: {report['document_count']}",
        f"- Chunk count: {report['chunk_count']}",
        f"- Question count: {report['question_count']}",
        "",
        "## Overall Metrics",
        "",
        f"- Recall@1: {format_percentage(metrics['recall_at_1'])}",
        f"- Recall@3: {format_percentage(metrics['recall_at_3'])}",
        f"- Recall@5: {format_percentage(metrics['recall_at_5'])}",
        f"- MRR: {metrics['mrr']:.4f}",
        f"- Zero-hit Rate: {format_percentage(metrics['zero_hit_rate'])}",
        f"- Average Retrieval Latency: {format_ms(metrics['average_retrieval_latency_ms'])}",
        f"- P95 Retrieval Latency: {format_ms(metrics['p95_retrieval_latency_ms'])}",
        "",
        "## Results by Difficulty",
        "",
        "| Difficulty | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for key, item in report["by_difficulty"].items():
        lines.append(
            f"| {key} | {item['count']} | {format_percentage(item['recall_at_1'])} | "
            f"{format_percentage(item['recall_at_3'])} | {format_percentage(item['recall_at_5'])} | "
            f"{item['mrr']:.4f} | {format_percentage(item['zero_hit_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Results by System",
            "",
            "| System | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in report["by_system"].items():
        lines.append(
            f"| {key} | {item['count']} | {format_percentage(item['recall_at_1'])} | "
            f"{format_percentage(item['recall_at_3'])} | {format_percentage(item['recall_at_5'])} | "
            f"{item['mrr']:.4f} | {format_percentage(item['zero_hit_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Results by Category",
            "",
            "| Category | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, item in report["by_category"].items():
        lines.append(
            f"| {key} | {item['count']} | {format_percentage(item['recall_at_1'])} | "
            f"{format_percentage(item['recall_at_3'])} | {format_percentage(item['recall_at_5'])} | "
            f"{item['mrr']:.4f} | {format_percentage(item['zero_hit_rate'])} |"
        )

    lines.extend(["", "## Failed Queries", ""])
    if not report["failed_queries"]:
        lines.append("No failed queries.")
    else:
        for result in report["failed_queries"]:
            top_sources = ", ".join(
                source["source"] for source in result["top_k_sources"]
            )
            lines.extend(
                [
                    f"### {result['id']}",
                    "",
                    f"- Question: {result['question']}",
                    f"- Expected source: {result['expected_source']}",
                    f"- Top-K sources: {top_sources or 'None'}",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Per-Question Results",
            "",
            "| ID | Expected Source | First Relevant Rank | Latency | Top-K Sources |",
            "|---|---|---:|---:|---|",
        ]
    )
    for result in report["results"]:
        first_rank = result["first_relevant_rank"] or "-"
        top_sources = ", ".join(
            source["source"] for source in result["top_k_sources"]
        )
        lines.append(
            f"| {result['id']} | {result['expected_source']} | {first_rank} | "
            f"{format_ms(result['latency_ms'])} | {top_sources} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_reports(report: dict[str, Any], reports_dir: Path) -> tuple[Path, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "v7_baseline_report.json"
    md_path = reports_dir / "v7_baseline_report.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return md_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Chroma dense retrieval baseline.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be >= 1")

    questions = load_eval_questions(args.questions)
    raw_documents = load_documents()
    if not raw_documents:
        raise ValueError("No documents loaded from data/raw. Run src/ingest.py first.")

    validate_expected_sources(questions, raw_documents)
    vectorstore = build_vectorstore()
    chunk_count = get_chroma_chunk_count(vectorstore)

    results = evaluate_questions(vectorstore, questions, top_k=args.top_k)
    report = build_evaluation_report(
        results=results,
        document_count=len(raw_documents),
        chunk_count=chunk_count,
    )
    md_path, json_path = write_reports(report, args.reports_dir)

    metrics = report["metrics"]
    print("Retrieval evaluation completed.")
    print(f"Questions: {report['question_count']}")
    print(f"Documents: {report['document_count']}")
    print(f"Chunks: {report['chunk_count']}")
    print(f"Recall@1: {format_percentage(metrics['recall_at_1'])}")
    print(f"Recall@3: {format_percentage(metrics['recall_at_3'])}")
    print(f"Recall@5: {format_percentage(metrics['recall_at_5'])}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"Zero-hit Rate: {format_percentage(metrics['zero_hit_rate'])}")
    print(f"Average Retrieval Latency: {format_ms(metrics['average_retrieval_latency_ms'])}")
    print(f"P95 Retrieval Latency: {format_ms(metrics['p95_retrieval_latency_ms'])}")
    print(f"Markdown report: {md_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
