"""Read the Gemini session cookies straight out of a signed-in browser.

Google wants the whole session cookie set now, not just `__Secure-1PSID` and `__Secure-1PSIDTS`,
so copying them by hand is a dozen values and one typo away from "signed out". This reads them
from a browser profile instead.

Firefox stores cookie values in clear, so it is the reliable source. Chromium-based browsers
encrypt them and, on macOS, bind the session to the device; `Device Bound Session Credentials`
then makes a copied cookie unusable elsewhere. Read the warning in the README before relying on
Chrome.

    python -m gemini2api.cookies --browser firefox
    python -m gemini2api.cookies --browser firefox --write   # replace config.yaml's accounts
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

# Every Google cookie is taken, not a chosen subset: probing showed that hand-picked sets which
# look sufficient (the PSID pair, or the classic SID family) are still bounced to ServiceLogin,
# while the whole jar the browser actually holds is accepted. Only analytics are dropped.
DROPPED = {"_ga", "_gcl_au", "_ga_WC57KJ50ZZ"}
GOOGLE_HOSTS = {".google.com", "accounts.google.com", "gemini.google.com"}


def firefox_roots() -> list[Path]:
    """Where Firefox keeps its profiles on this platform."""
    home = Path.home()
    if sys.platform == "darwin":
        return [home / "Library" / "Application Support" / "Firefox"]
    if sys.platform == "win32":
        appdata = Path(os.environ.get("APPDATA") or home / "AppData" / "Roaming")
        return [appdata / "Mozilla" / "Firefox"]
    # Debian/Fedora and the Snap package keep separate roots.
    return [
        home / ".mozilla" / "firefox",
        home / "snap" / "firefox" / "common" / ".mozilla" / "firefox",
    ]


def firefox_profiles(root: Path | None = None) -> list[Path]:
    """Every Firefox profile that holds a cookie database, newest last."""
    roots = [root] if root is not None else firefox_roots()
    profiles: list[Path] = []
    for base in roots:
        if not base.is_dir():
            continue
        # `Profiles/*` on every platform; a flat `*.default*` layout still shows up in
        # older Linux installs, so both are considered.
        profiles.extend(path.parent for path in base.glob("Profiles/*/cookies.sqlite"))
        profiles.extend(path.parent for path in base.glob("*.default*/cookies.sqlite"))
    unique = {path.resolve(): path for path in profiles}
    return sorted(unique.values(), key=lambda path: (path / "cookies.sqlite").stat().st_mtime)


def read_firefox(profile: Path) -> dict[str, str]:
    """The Gemini cookies in one Firefox profile.

    The file is copied first: Firefox holds it open while running, and reading a live SQLite
    database in place can see a half-written page.
    """
    source = profile / "cookies.sqlite"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "cookies.sqlite"
        shutil.copy(source, target)
        for suffix in ("-wal", "-shm"):
            sidecar = source.with_name(source.name + suffix)
            if sidecar.is_file():
                shutil.copy(sidecar, target.with_name(target.name + suffix))
        connection = sqlite3.connect(target)
        try:
            rows = connection.execute(
                "select host, name, value from moz_cookies where host in ({})".format(
                    ",".join("?" * len(GOOGLE_HOSTS))
                ),
                tuple(GOOGLE_HOSTS),
            ).fetchall()
        finally:
            connection.close()
    cookies: dict[str, str] = {}
    for _host, name, value in rows:
        if value and name not in DROPPED and not name.startswith("_ga"):
            cookies[name] = value
    return cookies


def pick_profile(explicit: Path | None = None) -> dict[str, Any]:
    """The profile that actually has a Gemini session, reported with which one it was."""
    candidates = [explicit] if explicit else firefox_profiles()
    if not candidates:
        raise SystemExit(
            "no Firefox profile with a cookie database was found. Install Firefox, sign in to "
            "gemini.google.com, and try again."
        )
    best: tuple[int, Path, dict[str, str]] | None = None
    for profile in candidates:
        cookies = read_firefox(profile)
        score = len(cookies)
        if best is None or score > best[0]:
            best = (score, profile, cookies)
    assert best is not None
    score, profile, cookies = best
    if "__Secure-1PSID" not in cookies:
        raise SystemExit(
            f"{profile} has no Gemini session cookies. Sign in to gemini.google.com in that "
            "Firefox profile first."
        )
    return {"profile": str(profile), "cookies": cookies, "count": score}


def write_config(cookies: dict[str, str], path: Path, proxy: str | None = None) -> None:
    """Replace the accounts in `path` with this one, keeping the rest of the file."""
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    extra = {
        name: value for name, value in cookies.items() if not name.startswith("__Secure-1PSID")
    }
    existing["accounts"] = [
        {
            "id": "firefox",
            "secure_1psid": cookies["__Secure-1PSID"],
            "secure_1psidts": cookies.get("__Secure-1PSIDTS", ""),
            "cookies": extra,
            **({"proxy": proxy} if proxy else {}),
        }
    ]
    path.write_text(yaml.safe_dump(existing, allow_unicode=True, sort_keys=False), encoding="utf-8")
    path.chmod(0o600)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default="firefox", choices=["firefox"])
    parser.add_argument("--profile", type=Path, default=None, help="an explicit profile directory")
    parser.add_argument("--write", action="store_true", help="write the cookies into config.yaml")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--proxy", default=None, help="proxy to record for this account")
    parser.add_argument("--json", action="store_true", help="print names and lengths only")
    args = parser.parse_args(argv)

    found = pick_profile(args.profile)
    cookies = found["cookies"]
    if args.json:
        print(
            json.dumps(
                {
                    "profile": found["profile"],
                    "cookies": {name: len(value) for name, value in sorted(cookies.items())},
                },
                indent=2,
            )
        )
    else:
        print(f"profile: {found['profile']}")
        for name, value in sorted(cookies.items()):
            print(f"  {name:24} {len(value):>4} bytes")
        if "__Secure-1PSIDTS" not in cookies:
            print(
                "warning: no __Secure-1PSIDTS; Google issues it on first use, but a session "
                "without it is usually stale"
            )

    if args.write:
        write_config(cookies, args.config, args.proxy)
        print(f"wrote {len(cookies)} cookie(s) into {args.config} (mode 600)")
    else:
        print("\nre-run with --write to put these into config.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
