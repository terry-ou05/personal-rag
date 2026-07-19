import json
from pathlib import Path

import pytest

from src.evaluate_retrieval import (
    calculate_latency_metrics,
    calculate_metrics,
    find_first_relevant_rank,
    load_eval_questions,
    normalize_source,
    validate_expected_sources,
)


def test_find_first_relevant_rank():
    retrieved = [
        {"source": "data/raw/a.md"},
        {"source": "data\\raw\\target.md"},
        {"source": "data/raw/other.md"},
    ]

    assert find_first_relevant_rank(retrieved, "target.md") == 2


def test_find_first_relevant_rank_empty_results():
    assert find_first_relevant_rank([], "target.md") is None


def test_calculate_recall_and_mrr():
    results = [
        {"first_relevant_rank": 1},
        {"first_relevant_rank": 3},
        {"first_relevant_rank": None},
    ]

    metrics = calculate_metrics(results)

    assert metrics["recall_at_1"] == pytest.approx(1 / 3)
    assert metrics["recall_at_3"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert metrics["zero_hit_rate"] == pytest.approx(1 / 3)


def test_calculate_latency_p95():
    metrics = calculate_latency_metrics([10, 20, 30, 40, 50])

    assert metrics["average_retrieval_latency_ms"] == pytest.approx(30)
    assert metrics["p95_retrieval_latency_ms"] == pytest.approx(50)


def test_normalize_source_handles_windows_paths():
    assert normalize_source("data\\raw\\itops_nginx_502.md") == "itops_nginx_502.md"
    assert normalize_source("data/raw/itops_nginx_502.md") == "itops_nginx_502.md"


def test_load_eval_questions_invalid_json(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{bad json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_eval_questions(path)


def test_load_eval_questions_missing_required_field(tmp_path: Path):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"id": "q1"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_eval_questions(path)


class FakeDocument:
    def __init__(self, source: str):
        self.metadata = {"source": source}


def test_validate_expected_source_missing():
    questions = [
        {
            "expected_source": "missing.md",
        }
    ]
    raw_documents = [FakeDocument("data/raw/existing.md")]

    with pytest.raises(ValueError, match="not in the current loaded documents"):
        validate_expected_sources(questions, raw_documents)
