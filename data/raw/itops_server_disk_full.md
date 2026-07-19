# Server Disk Space Is Full

## Scenario

A Linux server reports disk usage above 85% or 90%. Applications may fail to write logs, upload files, or create temporary files.

## Common Causes

- Application logs were not rotated.
- Temporary files accumulated under `/tmp`.
- Old deployment packages or backups were not cleaned.
- A database or message queue data directory grew unexpectedly.

## Troubleshooting Steps

1. Check disk usage:

```bash
df -h
```

2. Find large directories:

```bash
du -h --max-depth=1 /var | sort -h
```

3. Check large log files:

```bash
find /var/log -type f -size +100M -print
```

4. Clean only files that are confirmed safe, such as old rotated logs or temporary files.

5. If the data directory is growing quickly, check the related service before deleting files.

## Suggested Response

Confirm which mount point is full, locate the largest directories, clean safe old files, and add log rotation or retention policies to prevent recurrence.
