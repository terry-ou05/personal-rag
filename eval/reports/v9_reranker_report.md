# V9 Cross-Encoder Reranker Evaluation

- Evaluated at: 2026-07-19T16:04:58

## Dataset

- Documents: 10
- Chunks: 16
- Questions: 30

## Model Configuration

- Model name: BAAI/bge-reranker-base
- Device: cpu
- Candidate top-k: 5
- Final top-k: 5
- Batch size: 8

## Baseline Metrics

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 86.67% | 100.00% | 100.00% | 0.9278 | 0.00% | 9.98 ms | 10.38 ms |
| Hybrid RRF | 86.67% | 96.67% | 100.00% | 0.9122 | 0.00% | 10.35 ms | 11.34 ms |

## Reranker Metrics

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense + Reranker | 96.67% | 100.00% | 100.00% | 0.9833 | 0.00% | 760.23 ms | 793.29 ms |
| Hybrid + Reranker | 96.67% | 100.00% | 100.00% | 0.9833 | 0.00% | 747.28 ms | 780.22 ms |

## Candidate Recall

- Dense + Reranker Candidate Recall@5: 100.00%
- Hybrid + Reranker Candidate Recall@5: 100.00%

## Metric Delta

| Comparison | Recall@1 Delta | Recall@3 Delta | Recall@5 Delta | MRR Delta | Zero-hit Delta | Avg Latency Delta | P95 Latency Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense+Rerank vs Dense | +0.1000 | +0.0000 | +0.0000 | +0.0556 | +0.0000 | 750.25 ms | 782.91 ms |
| Hybrid+Rerank vs Hybrid | +0.1000 | +0.0333 | +0.0000 | +0.0711 | +0.0000 | 736.92 ms | 768.88 ms |
| Hybrid+Rerank vs Dense | +0.1000 | +0.0000 | +0.0000 | +0.0556 | +0.0000 | 737.30 ms | 769.84 ms |

## Cold and Warm Latency

### Dense + Reranker

- Model load time: 8201.15 ms
- Cold first query: 739.63 ms
- Warm average rerank latency: 749.12 ms
- Warm P95 rerank latency: 781.21 ms
- End-to-end average latency: 760.23 ms
- End-to-end P95 latency: 793.29 ms
- Average candidates reranked per query: 5.00

### Hybrid + Reranker

- Model load time: 8201.15 ms
- Cold first query: 752.75 ms
- Warm average rerank latency: 734.29 ms
- Warm P95 rerank latency: 766.04 ms
- End-to-end average latency: 747.28 ms
- End-to-end P95 latency: 780.22 ms
- Average candidates reranked per query: 5.00

## Non-Top-1 Analysis

| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |
|---|---|---:|---:|---:|---:|---|---|
| disk-002 | itops_server_disk_full.md | 2 | 1 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| disk-003 | itops_server_disk_full.md | 3 | 5 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| nginx-005 | itops_nginx_502.md | 2 | 3 | 2 | 2 | itops_server_cpu_high.md | itops_server_cpu_high.md |
| login-003 | itops_login_failure.md | 1 | 3 | 1 | 1 | itops_login_failure.md | itops_login_failure.md |
| login-005 | itops_login_failure.md | 2 | 2 | 1 | 1 | itops_login_failure.md | itops_login_failure.md |

## Improvements and Regressions

### Dense + Reranker Top-1 Improvements

| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |
|---|---|---:|---:|---:|---:|---|---|
| disk-002 | itops_server_disk_full.md | 2 | 1 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| disk-003 | itops_server_disk_full.md | 3 | 5 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| login-005 | itops_login_failure.md | 2 | 2 | 1 | 1 | itops_login_failure.md | itops_login_failure.md |

### Dense + Reranker Top-1 Regressions

| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |
|---|---|---:|---:|---:|---:|---|---|
| None | - | - | - | - | - | - | - |

### Rank Improvements

| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |
|---|---|---:|---:|---:|---:|---|---|
| disk-002 | itops_server_disk_full.md | 2 | 1 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| disk-003 | itops_server_disk_full.md | 3 | 5 | 1 | 1 | itops_server_disk_full.md | itops_server_disk_full.md |
| login-005 | itops_login_failure.md | 2 | 2 | 1 | 1 | itops_login_failure.md | itops_login_failure.md |

### Rank Regressions

| Query | Expected | Dense | Hybrid | Dense+Rerank | Hybrid+Rerank | Dense+Rerank Top-1 | Hybrid+Rerank Top-1 |
|---|---|---:|---:|---:|---:|---|---|
| None | - | - | - | - | - | - | - |

## Recommendation

Default to Dense + Reranker if the observed latency is acceptable for local use.

## Limitations

- Current dataset has only 10 documents, 16 chunks, and 30 evaluation queries.
- Test questions come from a small self-built IT operations corpus.
- Results do not represent a large enterprise knowledge base.
- Cross-Encoder reranking adds compute cost and model load time.
- This evaluates retrieval ranking only, not final answer generation quality.
- Query Rewrite, retrieval sufficiency checks, Qdrant, and LangGraph are not implemented in V9.
