"""Building a signed-in client from one account, in one place.

Two callers need this (the pool and `doctor`), and getting it wrong is not obvious: Google wants
the whole session cookie set now, so a client built from only `__Secure-1PSID` and
`__Secure-1PSIDTS` is redirected to ServiceLogin even while the browser session is valid.
"""

from __future__ import annotations

from gemini_webapi import GeminiClient

from .config import Account

# Google hands back `__Secure-1PSIDTS` on the first request and rotates it; the library persists
# the rotated values next to these names, so they must not be treated as caller-owned.
ROTATED = {"__Secure-1PSID", "__Secure-1PSIDTS"}


def make_client(account: Account) -> GeminiClient:
    """A client carrying every cookie this account owns, plus its proxy and fingerprint."""
    client = GeminiClient(
        secure_1psid=account.secure_1psid,
        secure_1psidts=account.secure_1psidts,
        proxy=account.proxy,
    )
    if account.impersonate:
        client.impersonate = account.impersonate
    for name, value in account.cookies.items():
        if name in ROTATED:
            continue
        client.cookies.set(name, value, domain=".google.com", secure=True)
    return client
