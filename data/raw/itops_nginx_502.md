# Nginx 502 Bad Gateway

## Scenario

Users see `502 Bad Gateway` from Nginx. This usually means Nginx cannot get a valid response from the upstream application service.

## Common Causes

- The upstream application process is down.
- The upstream port is wrong or not listening.
- The application is overloaded and closes connections.
- Nginx timeout settings are too low for the request.

## Troubleshooting Steps

1. Check Nginx error logs:

```bash
tail -n 100 /var/log/nginx/error.log
```

2. Confirm the upstream service is running:

```bash
systemctl status app-service
```

3. Check whether the upstream port is listening:

```bash
ss -lntp | grep 8080
```

4. Test the upstream directly from the Nginx server:

```bash
curl -v http://127.0.0.1:8080/health
```

5. If upstream is slow but alive, check application logs and consider timeout tuning.

## Suggested Response

For Nginx 502, first check whether the upstream application is alive and reachable. Then inspect Nginx and application logs to determine whether the issue is a crash, wrong port, overload, or timeout.
