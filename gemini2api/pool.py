"""A pool of signed-in Gemini web clients, plus the chat records that let a turn continue.

Gemini keeps a conversation server-side, so a later turn that extends the same history only
needs to send the new messages. That is what keeps a long agent loop cheap: without it every
turn resends the whole transcript.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from gemini_webapi import GeminiClient
from gemini_webapi.constants import AccountStatus
from loguru import logger

from .clients import make_client
from .config import Account, Settings


def history_key(model: str, messages: list[dict[str, Any]]) -> str:
    """A stable digest of one conversation as it stood after a turn.

    The digest covers the messages *a turn was answered from*, so a later request that replays
    those same messages plus new ones matches the chat that already holds that context. Matching
    is by longest prefix, because that is exactly the case: the same history, extended.
    """
    canonical = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": message.get("role"),
                    "content": message.get("content"),
                    "tool_calls": message.get("tool_calls"),
                    "tool_call_id": message.get("tool_call_id"),
                }
                for message in messages
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class StoredChat:
    """Where one conversation lives: which account opened it, and its native ids."""

    account_id: str
    metadata: list[str | None]
    model: str
    turns: int = 0
    last_used: float = field(default_factory=time.time)
    # How much of the incoming history this chat already had, set by `find_chat`.
    matched_messages: int = 0
    # Set when the chat was opened in auto_delete mode, so only those are ever deleted remotely.
    auto_delete: bool = False


class Pool:
    """Every configured account as a live client, handed out in turn."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._clients: dict[str, GeminiClient] = {}
        self._order: list[str] = []
        self._cursor = 0
        self._chats: dict[str, StoredChat] = {}
        self._locks: dict[str, asyncio.Lock] = {
            account.id: asyncio.Lock() for account in settings.accounts
        }
        self._deletions: list[tuple[str, str]] = []
        self._closed = False

    # -- lifecycle ------------------------------------------------------------------------

    async def start(self) -> None:
        for account in self.settings.accounts:
            self._clients[account.id] = make_client(account)
            self._order.append(account.id)
        await asyncio.gather(
            *(self._start_one(account) for account in self.settings.accounts),
            return_exceptions=True,
        )
        live = [cid for cid in self._order if self._healthy(cid)]
        if not live:
            raise RuntimeError(
                "no account signed in: every client fell back to a guest session. "
                "Check the __Secure-1PSID / __Secure-1PSIDTS cookies."
            )
        logger.info(f"accounts ready: {', '.join(live)}")

    async def _start_one(self, account: Account) -> bool:
        client = self._clients[account.id]
        try:
            await client.init(
                timeout=self.settings.timeout,
                auto_refresh=True,
                refresh_interval=self.settings.refresh_interval,
                verbose=False,
            )
        except Exception as exc:
            logger.error(f"account {account.id}: sign-in failed: {exc}")
            return False
        if not self._healthy(account.id):
            logger.error(
                f"account {account.id}: {client.account_status.name} - "
                f"{client.account_status.description}"
            )
            return False
        logger.info(f"account {account.id}: signed in, {len(client.list_models() or [])} model(s)")
        return True

    async def close(self) -> None:
        self._closed = True
        await asyncio.gather(
            *(client.close() for client in self._clients.values()), return_exceptions=True
        )

    def _healthy(self, account_id: str) -> bool:
        client = self._clients.get(account_id)
        if client is None:
            return False
        return client.account_status == AccountStatus.AVAILABLE

    # -- clients --------------------------------------------------------------------------

    def client(self, account_id: str) -> GeminiClient:
        return self._clients[account_id]

    def next_client(self) -> tuple[str, GeminiClient]:
        """The next signed-in account in rotation."""
        for _ in range(len(self._order)):
            account_id = self._order[self._cursor % len(self._order)]
            self._cursor += 1
            if self._healthy(account_id):
                return account_id, self._clients[account_id]
        raise RuntimeError("no signed-in account is available right now")

    def lock(self, account_id: str) -> asyncio.Lock:
        """Serialise turns on one account: its session ids are not safe to share."""
        return self._locks[account_id]

    # -- chat records ---------------------------------------------------------------------

    def find_chat(self, model: str, messages: list[dict[str, Any]]) -> StoredChat | None:
        """The chat that already holds the longest prefix of this history, if any.

        A later turn replays the transcript it was given plus what has happened since, so the
        stored digest is of the earlier, shorter list. Ties go to the longest match: that is the
        one carrying the most context, and therefore the fewest messages left to send.
        """
        for length in range(len(messages), 0, -1):
            key = history_key(model, messages[:length])
            chat = self._chats.get(key)
            if chat is None:
                continue
            if not self._healthy(chat.account_id):
                self._chats.pop(key, None)
                continue
            chat.last_used = time.time()
            chat.matched_messages = length
            return chat
        return None

    def store_chat(self, key: str, chat: StoredChat) -> None:
        self._chats[key] = chat
        # Bound memory the blunt way: the oldest untouched conversations go first.
        if len(self._chats) > 2048:
            oldest = sorted(self._chats.items(), key=lambda item: item[1].last_used)
            for stale_key, _ in oldest[: len(self._chats) - 2048]:
                self._chats.pop(stale_key, None)

    def drop_chat(self, key: str) -> None:
        self._chats.pop(key, None)

    def chat_count(self) -> int:
        return len(self._chats)

    def schedule_delete(self, account_id: str, cid: str) -> None:
        """Queue one Google-side chat for deletion; nothing is deleted while a turn is running."""
        self._deletions.append((account_id, cid))

    async def drain_deletions(self) -> int:
        """Delete the queued chats. Returns how many were actually removed."""
        if not self._deletions:
            return 0
        pending, self._deletions = self._deletions, []
        removed = 0
        for account_id, cid in pending:
            client = self._clients.get(account_id)
            if client is None or not self._healthy(account_id):
                continue
            try:
                await client.delete_chat(cid)
                removed += 1
            except Exception as exc:
                logger.warning(f"account {account_id}: could not delete chat {cid}: {exc}")
                self._deletions.append((account_id, cid))
        return removed

    async def reap_idle(self) -> int:
        """Expire idle chats; in auto_delete mode also remove them from the Google account."""
        ttl = self.settings.auto_delete_ttl
        now = time.time()
        expired = [(key, chat) for key, chat in self._chats.items() if now - chat.last_used > ttl]
        for key, chat in expired:
            self._chats.pop(key, None)
            cid = chat.metadata[0] if chat.metadata else None
            if chat.auto_delete and cid:
                self.schedule_delete(chat.account_id, cid)
        return len(expired)

    # -- read-only views ------------------------------------------------------------------

    def models(self) -> list[dict[str, Any]]:
        """Every model any signed-in account offers, once, with who can serve it."""
        seen: dict[str, dict[str, Any]] = {}
        for account_id in self._order:
            if not self._healthy(account_id):
                continue
            for model in self._clients[account_id].list_models() or []:
                if not model.is_available:
                    continue
                entry = seen.setdefault(
                    model.model_name,
                    {
                        "name": model.model_name,
                        "display_name": model.display_name,
                        "description": model.description,
                        "advanced": model.advanced_only,
                        "accounts": [],
                    },
                )
                entry["accounts"].append(account_id)
        return sorted(seen.values(), key=lambda item: item["name"])

    def quota(self) -> dict[str, Any]:
        """Per-account subscription state, quotas and usage, straight from Google."""
        report: dict[str, Any] = {}
        for account_id in self._order:
            if not self._healthy(account_id):
                report[account_id] = {"signed_in": False}
                continue
            client = self._clients[account_id]
            status = client.account_status
            usage = client.usage_info or {}
            # `usage_info.tier` is Google's own plan label (FREE / PRO / ULTRA); nothing here
            # infers a plan from which models happen to be listed.
            tier = usage.get("tier") or {}
            report[account_id] = {
                "signed_in": True,
                "status": status.name,
                "plan": tier.get("label"),
                "plan_id": tier.get("id"),
                # The binding window, then every window on its own.
                "windows": {
                    name: {
                        "usage_percentage": value.get("usage_percentage"),
                        "remaining_credits": value.get("remaining_credits"),
                        "reset_at": value.get("reset_at"),
                    }
                    for name, value in usage.items()
                    if isinstance(value, dict) and "window" in value
                },
                "quotas": client.quotas,
            }
        return report

    def health(self) -> dict[str, Any]:
        return {
            "accounts": {
                account_id: {
                    "signed_in": self._healthy(account_id),
                    "status": self._clients[account_id].account_status.name,
                }
                for account_id in self._order
            },
            "chats": len(self._chats),
            "closed": self._closed,
        }
