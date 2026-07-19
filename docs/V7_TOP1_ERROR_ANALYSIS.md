# V7 Top-1 Retrieval Error Analysis

This document analyzes the V7 dense retrieval baseline before starting V8 Hybrid Retrieval. It does not change the evaluation dataset, metrics, retriever code, or Streamlit workflow.

## Evaluation Summary

The analysis is based on `eval/reports/v7_baseline_report.json` from V7.

| Metric | Result |
|---|---:|
| Questions | 30 |
| Documents | 10 |
| Chunks | 16 |
| Top-1 Hits | 26 |
| Non-Top-1 Queries | 4 |
| Recall@1 | 86.67% |
| Recall@3 | 100.00% |
| Recall@5 | 100.00% |
| MRR | 0.9278 |
| Zero-hit Rate | 0.00% |

All four non-Top-1 queries still retrieved the expected source within Top-3. The main issue is therefore ranking precision, not complete retrieval failure.

## Non-Top-1 Query Analysis

| Query | Expected Source | Actual Rank | Dense Top-1 | Exact Keywords | BM25 Potential |
|---|---|---:|---|---|---|
| `disk-002` | `itops_server_disk_full.md` | 2 | `itops_login_failure.md` | Command `df -h`, percentage `90%`, root partition meaning | High |
| `disk-003` | `itops_server_disk_full.md` | 3 | `itops_server_cpu_high.md` | Log writing symptom, no exact command | Medium |
| `nginx-005` | `itops_nginx_502.md` | 2 | `itops_login_failure.md` | Gateway error, application process, logs | Medium |
| `login-005` | `itops_login_failure.md` | 2 | `itops_mysql_slow_query.md` | Account lock, password expiration | High |

### `disk-002`

- Question: `df -h 显示根分区 90%，下一步查哪里？`
- Difficulty: easy
- Query type: technical
- Expected source: `itops_server_disk_full.md`
- Expected source rank: 2
- Dense Top-1: `itops_login_failure.md`
- Top-5 sources: `itops_login_failure.md`, `itops_server_disk_full.md`, `my_notes.txt`, `itops_login_failure.md`, `itops_server_disk_full.md`

The question contains a concrete Linux command, `df -h`, plus an explicit disk-usage signal, `90%`. The expected disk document directly describes Linux disk usage above 85% or 90%, then lists `df -h` as the first troubleshooting command. The login runbook also contains Linux operational commands and troubleshooting steps, but its topic is authentication logs, account status, SSH keys, and network access.

Dense retrieval appears to treat this as a general Linux troubleshooting query and ranks another runbook-style document first. BM25 should have strong potential here because the command `df -h` is a precise lexical anchor that appears in the disk document and not in the login document.

BM25 Potential: High. The query has exact command and numeric disk-usage clues that lexical retrieval can reward.

### `disk-003`

- Question: `应用写日志失败，可能是哪个运维问题？`
- Difficulty: medium
- Query type: symptom
- Expected source: `itops_server_disk_full.md`
- Expected source rank: 3
- Dense Top-1: `itops_server_cpu_high.md`
- Top-5 sources: `itops_server_cpu_high.md`, `itops_login_failure.md`, `itops_server_disk_full.md`, `itops_nginx_502.md`, `itops_server_cpu_high.md`

The expected disk document says applications may fail to write logs, upload files, or create temporary files when disk usage is high. The CPU document also mentions application logs, retry loops, timeout errors, traffic spikes, slow API responses, and delayed background jobs. Both documents are server troubleshooting notes, so the semantic overlap is real.

This is a more ambiguous case than `disk-002`. The query does not include `df -h`, `/var/log`, disk, full, mount point, or a specific error. Dense Top-1 is understandable because application-level symptoms and logs also appear in the CPU incident document. BM25 may help if the exact phrase around log writing has stronger overlap with the disk document, but the query is short and symptom-based.

BM25 Potential: Medium. Lexical matching could help on the log-writing phrase, but the query lacks a strong disk-specific keyword.

### `nginx-005`

- Question: `网关错误但应用进程还在，接下来应该看哪些日志？`
- Difficulty: hard
- Query type: symptom
- Expected source: `itops_nginx_502.md`
- Expected source rank: 2
- Dense Top-1: `itops_login_failure.md`
- Top-5 sources: `itops_login_failure.md`, `itops_nginx_502.md`, `itops_server_cpu_high.md`, `itops_login_failure.md`, `itops_server_cpu_high.md`

The expected Nginx document covers `502 Bad Gateway`, upstream application process status, Nginx error logs, application logs, listening ports, and health checks. The login runbook also emphasizes checking logs and system access, but its scope is authentication: wrong password, expired credentials, locked accounts, SSH keys, auth service, firewall, and network policy.

The query says "gateway error" and "application process", but it does not mention `Nginx`, `502`, `upstream`, `error.log`, `8080`, or `curl`. Dense retrieval likely overweights the general "check logs" troubleshooting wording. The expected source is more specific, but the query is intentionally indirect.

BM25 Potential: Medium. It may help if "gateway" or "application process" aligns with the Nginx document, but adding explicit terms such as `502` or `Nginx` would make the lexical signal much stronger.

### `login-005`

- Question: `账号被锁定或者密码过期会导致什么现象？`
- Difficulty: medium
- Query type: symptom
- Expected source: `itops_login_failure.md`
- Expected source rank: 2
- Dense Top-1: `itops_mysql_slow_query.md`
- Top-5 sources: `itops_mysql_slow_query.md`, `itops_login_failure.md`, `itops_login_failure.md`, `itops_redis_memory_high.md`, `itops_server_cpu_high.md`

The expected login runbook directly lists wrong password, expired credentials, account lock after repeated failures, and users being unable to log in. The MySQL document discusses slow queries, API latency, timeouts, SQL execution plans, indexes, process lists, and transactions. It does not describe account lock or password expiration as an operational login failure.

This looks like a clear ranking error. The query contains exact authentication concepts that map directly to the login document. BM25 should help because `账号`, `锁定`, `密码`, and `过期` are highly discriminative for the login runbook compared with the MySQL slow query note.

BM25 Potential: High. The query has direct lexical signals for the expected source, and Dense Top-1 is less topically specific.

## Overall Findings

- The V7 dense retriever already has complete Top-3 and Top-5 recall on this dataset, so V8 should focus on Top-1 ranking quality and MRR rather than broad recall.
- Two cases have strong exact-match signals: `disk-002` with `df -h`, and `login-005` with account-lock and password-expiration terms. These are good candidates for BM25 or Hybrid Retrieval improvement.
- Two cases are more symptom-oriented: `disk-003` and `nginx-005`. BM25 may help less unless the query contains stronger technical terms.
- No complete zero-hit failures were found.
- `disk-003` has mild evaluation ambiguity because "application logs" also appears in other troubleshooting contexts, although the disk document is still the best expected source because it explicitly mentions applications failing to write logs.
- `nginx-005` is intentionally indirect. The expected source is reasonable, but exact wording such as `Nginx` or `502` would make the query less ambiguous.
- The dataset is small: 30 questions over a mixed knowledge base of 10 documents and 16 chunks. Results are useful as a development baseline, but they should not be treated as a broad production benchmark.

## V8 Hypothesis

V8 should compare the existing dense retriever against a hybrid retriever that combines dense semantic search with a lexical signal such as BM25.

Expected improvements:

- `disk-002` and `login-005` are likely to improve because they contain strong exact keywords.
- `disk-003` and `nginx-005` may improve only if the lexical retriever captures the right operational terms.
- Recall@3 and Recall@5 are already saturated at 100.00%, so V8 should focus on Recall@1, MRR, regression count, and retrieval latency.
- Hybrid Retrieval should not be assumed better by default. It must be accepted only if the actual V8 evaluation improves ranking quality without causing unacceptable latency or regressions.
