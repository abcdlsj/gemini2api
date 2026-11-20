"""Keys minted per client, so one caller can be revoked without changing everyone else.

The static `api_key` in the config stays what it always was: the operator's own key. It is also
the credential a client presents once before it can mint a key of its own, which is what keeps
`127.0.0.1` from being a way for any local process — or any page that can reach loopback — to
hand itself a key.

Minted keys live next to the rotating cookies (`state_dir/keys.json`, mode 600) rather than in
the config file, so a gateway can grant one without rewriting a file the operator edits by hand.
Only the SHA-256 of a key is stored: the plaintext is returned once, at mint time.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

KEYS_FILE = "keys.json"


def digest(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class KeyStore:
    """Minted keys, read at startup and updated on mint or revoke."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._entries: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            self._entries = []
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8") or "{}")
        except (OSError, ValueError):
            # A corrupt key file must not stop the server: the configured api_key still works.
            self._entries = []
            return
        entries = raw.get("keys") if isinstance(raw, dict) else None
        self._entries = [
            entry
            for entry in (entries or [])
            if isinstance(entry, dict) and isinstance(entry.get("sha256"), str)
        ]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"keys": self._entries}, indent=2) + "\n"
        # Atomic: a half-written file would lock every minted client out.
        with tempfile.NamedTemporaryFile(
            "w", dir=self.path.parent, delete=False, encoding="utf-8"
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            staged = Path(handle.name)
        staged.chmod(0o600)
        os.replace(staged, self.path)

    def __bool__(self) -> bool:
        return bool(self._entries)

    def verify(self, token: str) -> bool:
        supplied = digest(token)
        return any(
            secrets.compare_digest(entry["sha256"], supplied) for entry in self._entries
        )

    def mint(self, label: str | None = None) -> dict[str, Any]:
        """A new key. The plaintext appears in the return value and is never stored."""
        key = "g2k-" + secrets.token_urlsafe(32)
        entry = {
            "id": secrets.token_hex(8),
            "label": (label or "client").strip()[:64] or "client",
            "sha256": digest(key),
            "prefix": key[:12],
            "created_at": int(time.time()),
        }
        self._entries.append(entry)
        self._flush()
        return {**self.public(entry), "key": key}

    def keys(self) -> list[dict[str, Any]]:
        return [self.public(entry) for entry in self._entries]

    def revoke(self, key_id: str) -> bool:
        kept = [entry for entry in self._entries if entry.get("id") != key_id]
        if len(kept) == len(self._entries):
            return False
        self._entries = kept
        self._flush()
        return True

    @staticmethod
    def public(entry: dict[str, Any]) -> dict[str, Any]:
        return {name: entry.get(name) for name in ("id", "label", "prefix", "created_at")}
