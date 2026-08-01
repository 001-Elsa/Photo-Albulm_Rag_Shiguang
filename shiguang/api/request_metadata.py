from __future__ import annotations

from ipaddress import ip_address

from fastapi import Request


def client_ip(request: Request) -> str | None:
    """Return a normalized client IP suitable for PostgreSQL ``inet`` columns.

    Test clients and some non-HTTP transports use symbolic peer names such as
    ``testclient``.  They are useful for rate-limit keys but cannot be stored
    in an ``inet`` column, so retain only syntactically valid IP addresses.
    """
    forwarded = request.headers.get("X-Forwarded-For", "").split(",", 1)[0].strip()
    candidate = forwarded or (request.client.host if request.client else "")
    try:
        return str(ip_address(candidate))
    except ValueError:
        return None
