# V7 Retrieval Evaluation Baseline

- Evaluated at: 2026-07-19T14:43:53
- Document count: 10
- Chunk count: 16
- Question count: 30

## Overall Metrics

- Recall@1: 86.67%
- Recall@3: 100.00%
- Recall@5: 100.00%
- MRR: 0.9278
- Zero-hit Rate: 0.00%
- Average Retrieval Latency: 9.81 ms
- P95 Retrieval Latency: 10.35 ms

## Results by Difficulty

| Difficulty | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |
|---|---:|---:|---:|---:|---:|---:|
| easy | 13 | 92.31% | 100.00% | 100.00% | 0.9615 | 0.00% |
| hard | 6 | 83.33% | 100.00% | 100.00% | 0.9167 | 0.00% |
| medium | 11 | 81.82% | 100.00% | 100.00% | 0.8939 | 0.00% |

## Results by System

| System | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |
|---|---:|---:|---:|---:|---:|---:|
| authentication | 5 | 80.00% | 100.00% | 100.00% | 0.9000 | 0.00% |
| mysql | 5 | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% |
| nginx | 5 | 80.00% | 100.00% | 100.00% | 0.9000 | 0.00% |
| redis | 5 | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% |
| server | 10 | 80.00% | 100.00% | 100.00% | 0.8833 | 0.00% |

## Results by Category

| Category | Count | Recall@1 | Recall@3 | Recall@5 | MRR | Zero-hit |
|---|---:|---:|---:|---:|---:|---:|
| application | 5 | 80.00% | 100.00% | 100.00% | 0.9000 | 0.00% |
| cache | 5 | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% |
| database | 5 | 100.00% | 100.00% | 100.00% | 1.0000 | 0.00% |
| infrastructure | 10 | 80.00% | 100.00% | 100.00% | 0.8833 | 0.00% |
| security | 5 | 80.00% | 100.00% | 100.00% | 0.9000 | 0.00% |

## Failed Queries

No failed queries.

## Per-Question Results

| ID | Expected Source | First Relevant Rank | Latency | Top-K Sources |
|---|---|---:|---:|---|
| cpu-001 | itops_server_cpu_high.md | 1 | 29.18 ms | itops_server_cpu_high.md, itops_server_disk_full.md, itops_login_failure.md, itops_nginx_502.md, itops_server_cpu_high.md |
| cpu-002 | itops_server_cpu_high.md | 1 | 8.79 ms | itops_server_cpu_high.md, itops_server_disk_full.md, itops_nginx_502.md, itops_server_cpu_high.md, itops_nginx_502.md |
| cpu-003 | itops_server_cpu_high.md | 1 | 9.35 ms | itops_server_cpu_high.md, itops_nginx_502.md, itops_server_disk_full.md, itops_nginx_502.md, itops_login_failure.md |
| cpu-004 | itops_server_cpu_high.md | 1 | 8.82 ms | itops_server_cpu_high.md, itops_server_cpu_high.md, itops_mysql_slow_query.md, itops_server_disk_full.md, itops_nginx_502.md |
| cpu-005 | itops_server_cpu_high.md | 1 | 9.05 ms | itops_server_cpu_high.md, itops_server_disk_full.md, itops_login_failure.md, itops_nginx_502.md, itops_redis_memory_high.md |
| disk-001 | itops_server_disk_full.md | 1 | 9.41 ms | itops_server_disk_full.md, itops_server_disk_full.md, itops_server_cpu_high.md, itops_redis_memory_high.md, itops_nginx_502.md |
| disk-002 | itops_server_disk_full.md | 2 | 9.93 ms | itops_login_failure.md, itops_server_disk_full.md, my_notes.txt, itops_login_failure.md, itops_server_disk_full.md |
| disk-003 | itops_server_disk_full.md | 3 | 8.54 ms | itops_server_cpu_high.md, itops_login_failure.md, itops_server_disk_full.md, itops_nginx_502.md, itops_server_cpu_high.md |
| disk-004 | itops_server_disk_full.md | 1 | 9.77 ms | itops_server_disk_full.md, itops_server_cpu_high.md, itops_login_failure.md, itops_server_disk_full.md, itops_mysql_slow_query.md |
| disk-005 | itops_server_disk_full.md | 1 | 8.83 ms | itops_server_disk_full.md, itops_server_disk_full.md, itops_login_failure.md, itops_server_cpu_high.md, itops_mysql_slow_query.md |
| nginx-001 | itops_nginx_502.md | 1 | 9.33 ms | itops_nginx_502.md, itops_nginx_502.md, itops_login_failure.md, itops_redis_memory_high.md, itops_server_disk_full.md |
| nginx-002 | itops_nginx_502.md | 1 | 10.35 ms | itops_nginx_502.md, itops_nginx_502.md, itops_login_failure.md, itops_server_disk_full.md, itops_login_failure.md |
| nginx-003 | itops_nginx_502.md | 1 | 8.70 ms | itops_nginx_502.md, itops_nginx_502.md, itops_server_cpu_high.md, itops_login_failure.md, itops_server_disk_full.md |
| nginx-004 | itops_nginx_502.md | 1 | 9.75 ms | itops_nginx_502.md, itops_login_failure.md, itops_nginx_502.md, itops_redis_memory_high.md, itops_redis_memory_high.md |
| nginx-005 | itops_nginx_502.md | 2 | 9.77 ms | itops_login_failure.md, itops_nginx_502.md, itops_server_cpu_high.md, itops_login_failure.md, itops_server_cpu_high.md |
| mysql-001 | itops_mysql_slow_query.md | 1 | 8.84 ms | itops_mysql_slow_query.md, itops_mysql_slow_query.md, itops_server_cpu_high.md, itops_server_disk_full.md, itops_server_disk_full.md |
| mysql-002 | itops_mysql_slow_query.md | 1 | 9.99 ms | itops_mysql_slow_query.md, itops_server_cpu_high.md, itops_mysql_slow_query.md, itops_server_disk_full.md, itops_server_disk_full.md |
| mysql-003 | itops_mysql_slow_query.md | 1 | 8.89 ms | itops_mysql_slow_query.md, itops_mysql_slow_query.md, itops_redis_memory_high.md, itops_server_disk_full.md, itops_server_disk_full.md |
| mysql-004 | itops_mysql_slow_query.md | 1 | 8.52 ms | itops_mysql_slow_query.md, itops_server_disk_full.md, itops_server_cpu_high.md, itops_server_disk_full.md, itops_mysql_slow_query.md |
| mysql-005 | itops_mysql_slow_query.md | 1 | 9.17 ms | itops_mysql_slow_query.md, itops_login_failure.md, itops_mysql_slow_query.md, itops_server_cpu_high.md, itops_redis_memory_high.md |
| redis-001 | itops_redis_memory_high.md | 1 | 9.36 ms | itops_redis_memory_high.md, itops_redis_memory_high.md, itops_server_cpu_high.md, itops_server_disk_full.md, itops_mysql_slow_query.md |
| redis-002 | itops_redis_memory_high.md | 1 | 9.36 ms | itops_redis_memory_high.md, itops_redis_memory_high.md, itops_nginx_502.md, itops_server_cpu_high.md, itops_server_disk_full.md |
| redis-003 | itops_redis_memory_high.md | 1 | 8.45 ms | itops_redis_memory_high.md, itops_redis_memory_high.md, itops_server_disk_full.md, itops_server_disk_full.md, itops_mysql_slow_query.md |
| redis-004 | itops_redis_memory_high.md | 1 | 8.85 ms | itops_redis_memory_high.md, itops_redis_memory_high.md, itops_server_cpu_high.md, itops_login_failure.md, itops_server_disk_full.md |
| redis-005 | itops_redis_memory_high.md | 1 | 8.68 ms | itops_redis_memory_high.md, itops_redis_memory_high.md, itops_nginx_502.md, itops_server_cpu_high.md, itops_nginx_502.md |
| login-001 | itops_login_failure.md | 1 | 8.50 ms | itops_login_failure.md, itops_login_failure.md, itops_server_disk_full.md, itops_server_cpu_high.md, itops_nginx_502.md |
| login-002 | itops_login_failure.md | 1 | 9.10 ms | itops_login_failure.md, itops_login_failure.md, itops_server_disk_full.md, itops_nginx_502.md, itops_nginx_502.md |
| login-003 | itops_login_failure.md | 1 | 8.51 ms | itops_login_failure.md, itops_server_disk_full.md, itops_nginx_502.md, itops_login_failure.md, itops_server_cpu_high.md |
| login-004 | itops_login_failure.md | 1 | 8.92 ms | itops_login_failure.md, itops_login_failure.md, itops_server_disk_full.md, itops_mysql_slow_query.md, itops_server_disk_full.md |
| login-005 | itops_login_failure.md | 2 | 9.53 ms | itops_mysql_slow_query.md, itops_login_failure.md, itops_login_failure.md, itops_redis_memory_high.md, itops_server_cpu_high.md |
