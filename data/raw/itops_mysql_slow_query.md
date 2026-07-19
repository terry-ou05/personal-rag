# MySQL Slow Query Troubleshooting

## Scenario

Database queries become slow, causing API latency or timeout issues. The application may show increased response time during peak traffic.

## Common Causes

- Missing index on frequently filtered columns.
- SQL scans too many rows.
- Lock wait or long-running transaction.
- Sudden traffic increase or inefficient pagination.

## Troubleshooting Steps

1. Enable or inspect slow query logs.

2. Find slow SQL statements and check execution plans:

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 123;
```

3. Check whether important filter and join columns have indexes.

4. Inspect active queries:

```sql
SHOW PROCESSLIST;
```

5. If lock wait is suspected, identify long transactions before killing sessions.

## Suggested Response

Start from slow query logs and execution plans. Confirm whether indexes are used, whether full table scans happen, and whether lock waits or long transactions are causing delays.
