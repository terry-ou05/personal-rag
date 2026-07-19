# Redis Memory Usage Is High

## Scenario

Redis memory usage keeps increasing and approaches the configured memory limit. The application may see cache write failures, evictions, or higher latency.

## Common Causes

- Large keys or big hash/list/set structures.
- Cache keys do not have TTL.
- Traffic increase creates too many cached objects.
- Eviction policy is not suitable for the workload.

## Troubleshooting Steps

1. Check Redis memory information:

```bash
redis-cli INFO memory
```

2. Check key count and TTL patterns:

```bash
redis-cli DBSIZE
redis-cli --scan | head
```

3. Look for big keys:

```bash
redis-cli --bigkeys
```

4. Confirm whether important cache keys have expiration time.

5. Review `maxmemory` and `maxmemory-policy` settings before changing them.

## Suggested Response

Investigate big keys, missing TTL, and memory policy first. Avoid flushing Redis directly unless the business impact is clearly understood.
