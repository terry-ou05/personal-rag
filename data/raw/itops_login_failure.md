# User Login Failure Runbook

## Scenario

Users report that they cannot log in to a server or internal system. The issue may affect one user, one group, or many users.

## Common Causes

- Wrong password or expired credential.
- Account is locked after repeated failures.
- SSH key or permission is incorrect.
- Authentication service is unavailable.
- Firewall or network policy blocks access.

## Troubleshooting Steps

1. Confirm the affected scope: one user, one host, or many users.

2. Check authentication logs:

```bash
tail -n 100 /var/log/auth.log
```

3. Check whether the account is locked or expired.

4. If SSH key login is used, verify file permissions:

```bash
ls -la ~/.ssh
```

5. Check network access to the target host and authentication service.

## Suggested Response

Start by confirming the affected scope and checking authentication logs. If only one user is affected, inspect account status and key permissions. If many users are affected, check the authentication service and network path.
