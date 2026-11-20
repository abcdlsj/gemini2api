"""Configuration: YAML file, with every value overridable by a G2A_ prefixed env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "config.yaml"


def _env(name: str, default: Any, cast: type = str) -> Any:
    """A G2A_ prefixed environment variable, or the default."""
    raw = os.getenv(f"G2A_{name}")
    if raw is None or raw == "":
        return default
    if cast is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if cast is int:
        return int(raw)
    return raw


def parse_cookie_string(raw: str) -> dict[str, str]:
    """`name=value; name=value` as a mapping, which is how a browser hands cookies over."""
    cookies: dict[str, str] = {}
    for part in raw.split(";"):
        name, _, value = part.strip().partition("=")
        if name and value:
            cookies[name] = value
    return cookies


@dataclass
class Account:
    """One signed-in Gemini web account.

    Google now wants the whole session cookie set, not only the two PSID cookies: a request
    carrying just `__Secure-1PSID`/`__Secure-1PSIDTS` is bounced to ServiceLogin even while the
    browser session is valid. Anything else the account needs travels in `cookies`.
    """

    id: str
    secure_1psid: str
    secure_1psidts: str | None = None
    proxy: str | None = None
    impersonate: str | None = None
    cookies: dict[str, str] = field(default_factory=dict)

    def all_cookies(self) -> dict[str, str]:
        """Every cookie this account sends, the two PSID ones included."""
        merged = dict(self.cookies)
        merged["__Secure-1PSID"] = self.secure_1psid
        if self.secure_1psidts:
            merged["__Secure-1PSIDTS"] = self.secure_1psidts
        return merged


@dataclass
class Settings:
    """Everything the server needs, resolved from file and environment."""

    accounts: list[Account] = field(default_factory=list)
    api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = 8790
    # A chat that stays in the Google account keeps context across turns; deleting it once idle
    # stops the gateway from filling the account history. "temporary" never persists but Google
    # may close the window at any time and silently drop context.
    chat_mode: str = "auto_delete"
    auto_delete_ttl: int = 86400
    default_thinking: bool = False
    max_prompt_chars: int = 900_000
    # How long to wait on Google before giving up on one turn.
    timeout: float = 450.0
    refresh_interval: float = 600.0
    # Where `gemini-webapi` keeps the cookies it rotates. It defaults to a temp directory, where a
    # stale copy shadows the configured cookies and turns a valid session into a guest session;
    # a stable path makes that cache predictable and lets it be deleted deliberately.
    state_dir: str = ".gemini2api-state"

    @property
    def temporary(self) -> bool:
        return self.chat_mode == "temporary"

    @property
    def keys_path(self) -> Path:
        """Minted keys sit with the rotating cookies, not in the hand-edited config."""
        return Path(self.state_dir) / "keys.json"

    def account(self, account_id: str) -> Account:
        for account in self.accounts:
            if account.id == account_id:
                return account
        raise KeyError(account_id)


def _account_from(raw: dict[str, Any], index: int) -> Account:
    account_id = str(raw.get("id") or f"account-{index + 1}")
    psid = raw.get("secure_1psid")
    if not psid:
        raise ValueError(f"account {account_id!r}: secure_1psid is required")
    extra = raw.get("cookies") or {}
    if isinstance(extra, str):
        extra = parse_cookie_string(extra)
    if not isinstance(extra, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in extra.items()
    ):
        raise ValueError(f"account {account_id!r}: cookies must be a mapping of name to value")
    return Account(
        id=account_id,
        secure_1psid=str(psid),
        secure_1psidts=raw.get("secure_1psidts") or None,
        proxy=raw.get("proxy") or None,
        impersonate=raw.get("impersonate") or None,
        cookies=dict(extra),
    )


def load(path: str | Path | None = None) -> Settings:
    """Read the config file if there is one, then let the environment win."""
    config_path = Path(path or os.getenv("G2A_CONFIG") or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = {}
    if config_path.is_file():
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: expected a mapping at the top level")

    accounts_raw = raw.get("accounts") or []
    if not isinstance(accounts_raw, list):
        raise ValueError("accounts must be a list")
    accounts = [_account_from(item, index) for index, item in enumerate(accounts_raw)]

    # A single account can come entirely from the environment, which keeps cookies out of files.
    env_psid = os.getenv("G2A_SECURE_1PSID")
    if env_psid:
        accounts = [
            Account(
                id=os.getenv("G2A_ACCOUNT_ID") or "env",
                secure_1psid=env_psid,
                secure_1psidts=os.getenv("G2A_SECURE_1PSIDTS") or None,
                proxy=os.getenv("G2A_PROXY") or None,
                impersonate=os.getenv("G2A_IMPERSONATE") or None,
                cookies=parse_cookie_string(os.getenv("G2A_COOKIES") or ""),
            )
        ]
    if not accounts:
        raise ValueError(
            "no accounts configured: set accounts[].secure_1psid in the config file, "
            "or export G2A_SECURE_1PSID"
        )

    settings = Settings(
        accounts=accounts,
        api_key=_env("API_KEY", raw.get("api_key"), str) or None,
        host=_env("HOST", raw.get("host", "127.0.0.1"), str),
        port=_env("PORT", raw.get("port", 8790), int),
        chat_mode=_env("CHAT_MODE", raw.get("chat_mode", "auto_delete"), str),
        auto_delete_ttl=_env("AUTO_DELETE_TTL", raw.get("auto_delete_ttl", 86400), int),
        default_thinking=_env("DEFAULT_THINKING", raw.get("default_thinking", False), bool),
        max_prompt_chars=_env("MAX_PROMPT_CHARS", raw.get("max_prompt_chars", 900_000), int),
        timeout=float(_env("TIMEOUT", raw.get("timeout", 450.0), str)),
        refresh_interval=float(_env("REFRESH_INTERVAL", raw.get("refresh_interval", 600.0), str)),
        state_dir=_env("STATE_DIR", raw.get("state_dir", ".gemini2api-state"), str),
    )
    if settings.chat_mode not in {"normal", "temporary", "auto_delete"}:
        raise ValueError(
            f"chat_mode must be normal, temporary or auto_delete, got {settings.chat_mode!r}"
        )
    return settings
