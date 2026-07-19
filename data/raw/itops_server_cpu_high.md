# Server CPU Usage Is Too High

## Scenario

A Linux server reports sustained CPU usage above 90%. Users may notice slow API responses, delayed background jobs, or timeout errors.

## Common Causes

- One process is consuming most CPU resources.
- Too many worker processes are running at the same time.
- A scheduled task or batch job is running during peak traffic.
- The server is under abnormal traffic or repeated retry requests.

## Troubleshooting Steps

1. Check system load:

```bash
uptime
top
```

2. Identify the highest CPU processes:

```bash
ps aux --sort=-%cpu | head
```

3. Check whether the process is expected. If it is a known batch job, confirm its schedule and business impact.

4. Check application logs for retry loops, timeout errors, or traffic spikes.

5. If the process is abnormal, restart the related service after confirming impact with the team.

## Suggested Response

Start by identifying the top CPU process, then decide whether it is normal business load, a scheduled job, or an abnormal process. Avoid killing processes directly before confirming service impact.
