# Security notes

## Secret handling

- Runtime data under `data/` is ignored and must never be committed.
- Compose secrets live in `deploy/secrets/*.txt` (gitignored). Only `*.example` placeholders are tracked.
- Enterprise mode injects secrets via environment variables or `*_FILE` mounts.
- `Config.save()` excludes database, MinIO and metrics secrets from `data/config.json`.

## History cleanup

Old commits previously contained SQLite databases, logs, `secret.key` and
`admin_initial_password.txt`. Local history was rewritten with `git filter-repo`.

If the public remote still has the old history, force-push the cleaned history and
treat every previously committed secret as compromised:

1. Rotate session secrets and admin passwords.
2. Recreate Postgres / MinIO / Grafana credentials.
3. Ask collaborators to re-clone instead of pulling.

## CI controls

- Gitleaks scans the repository history on every push/PR.
- `pip-audit` checks core, enterprise and development dependency sets.
- Dependabot watches pip, GitHub Actions and Docker ecosystems.

These controls reduce recurrence risk; they do not replace secret rotation after a leak.
