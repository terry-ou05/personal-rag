# V8 Hybrid Retrieval Report

- Evaluated at: 2026-07-19T15:28:20

## Dataset

- Documents: 10
- Chunks: 16
- Questions: 30

## Configuration

- Dense top-k: 10
- BM25 top-k: 10
- Final top-k: 5
- rrf_k: 60
- Tokenizer: jieba for Chinese terms plus regex extraction for IT tokens, commands, paths, numbers, IPs, and dotted/underscore/hyphen words.

## Metrics Comparison

| Mode | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit | Avg Latency | P95 Latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 86.67% | 100.00% | 100.00% | 0.9278 | 0.00% | 10.24 ms | 10.65 ms |
| BM25 | 53.33% | 70.00% | 70.00% | 0.6167 | 30.00% | 0.25 ms | 0.33 ms |
| Hybrid RRF | 86.67% | 96.67% | 100.00% | 0.9122 | 0.00% | 12.16 ms | 13.98 ms |

## Delta vs V7

Dense in this report uses the same expected-source matching rule as V7. Delta rows are measured against the Dense result from this V8 run.

| Mode | Recall@1 Delta | Recall@3 Delta | Recall@5 Delta | MRR Delta | Zero-hit Delta | Avg Latency Delta | P95 Latency Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 - Dense | -0.3333 | -0.3000 | -0.3000 | -0.3111 | +0.3000 | -9.99 ms | -10.32 ms |
| Hybrid - Dense | +0.0000 | -0.0333 | +0.0000 | -0.0156 | +0.0000 | 1.92 ms | 3.33 ms |

## Non-Top-1 Query Analysis

| Query | Expected Source | Dense Rank | BM25 Rank | Hybrid Rank | Dense Top-1 | BM25 Top-1 | Hybrid Top-1 |
|---|---|---:|---:|---:|---|---|---|
| disk-002 | itops_server_disk_full.md | 2 | 1 | 1 | itops_login_failure.md | itops_server_disk_full.md | itops_server_disk_full.md |
| disk-003 | itops_server_disk_full.md | 3 | - | 5 | itops_server_cpu_high.md | my_notes.txt | itops_server_cpu_high.md |
| nginx-005 | itops_nginx_502.md | 2 | - | 3 | itops_login_failure.md | my_notes.txt | itops_login_failure.md |
| login-005 | itops_login_failure.md | 2 | - | 2 | itops_mysql_slow_query.md | - | itops_mysql_slow_query.md |

The four focus queries are the V7 non-Top-1 cases. They show whether lexical matching and RRF changed the ranking of known dense weaknesses.

## Improvements and Regressions

### Top-1 Improvements

| Query | Expected Source | Dense Rank | BM25 Rank | Hybrid Rank | Dense Top-1 | BM25 Top-1 | Hybrid Top-1 |
|---|---|---:|---:|---:|---|---|---|
| disk-002 | itops_server_disk_full.md | 2 | 1 | 1 | itops_login_failure.md | itops_server_disk_full.md | itops_server_disk_full.md |

### Top-1 Regressions

| Query | Expected Source | Dense Rank | BM25 Rank | Hybrid Rank | Dense Top-1 | BM25 Top-1 | Hybrid Top-1 |
|---|---|---:|---:|---:|---|---|---|
| login-003 | itops_login_failure.md | 1 | - | 3 | itops_login_failure.md | itops_server_cpu_high.md | itops_server_disk_full.md |

## Complementarity Analysis

- Dense-only hits: 9
- BM25-only hits: 0
- Hybrid recovered queries: 0
- Unchanged Top-1 status queries: 28

BM25 is expected to help most when a query contains exact commands, error codes, file names, service names, or configuration keys. Dense retrieval remains useful for short symptom descriptions where wording differs from the document.

## Recommendation

Default to Dense and keep Hybrid RRF optional because Hybrid introduced Top-1 regressions.

## Limitations

- Current dataset has only 10 documents, 16 chunks, and 30 evaluation questions.
- Evaluation questions come from a small self-built IT operations corpus.
- Results do not represent a large enterprise knowledge base.
- Recall@3 and Recall@5 were already saturated in V7.
- The main decision signals for V8 are Recall@1, MRR, regressions, and latency.
