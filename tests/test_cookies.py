"""Cross-platform Firefox cookie discovery and live-WAL reading."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from gemini2api import cookies as cookies_mod


def _make_profile(path: Path, rows: list[tuple[str, str, str]], *, wal: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path / "cookies.sqlite")
    if wal:
        db.execute("PRAGMA journal_mode = WAL")
    db.execute("CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT)")
    db.executemany("INSERT INTO moz_cookies VALUES (?, ?, ?)", rows)
    db.commit()
    if not wal:
        db.close()
        return path
    return path


def test_firefox_profiles_finds_nested_and_custom_flat_profiles(tmp_path: Path) -> None:
    nested = _make_profile(tmp_path / "Profiles" / "abc.default-release", [])
    flat_custom = _make_profile(tmp_path / "xyz.work", [])

    found = cookies_mod.firefox_profiles(tmp_path)
    assert set(found) == {nested, flat_custom}


def test_read_firefox_reads_open_wal_database(tmp_path: Path) -> None:
    profile = tmp_path / "live.default"
    profile.mkdir()
    conn = sqlite3.connect(profile / "cookies.sqlite")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT)")
        conn.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?)",
            (".google.com", "__Secure-1PSID", "psid-live"),
        )
        conn.execute(
            "INSERT INTO moz_cookies VALUES (?, ?, ?)",
            (".google.com", "_ga_ABC123", "drop-me"),
        )
        conn.commit()

        jar = cookies_mod.read_firefox(profile)
        assert jar == {"__Secure-1PSID": "psid-live"}
    finally:
        conn.close()


def test_pick_profile_prefers_signed_in_profile_and_skips_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logged_out = _make_profile(
        tmp_path / "Profiles" / "a.default",
        [
            (".google.com", "NID", "1"),
            (".google.com", "AEC", "2"),
            (".google.com", "SOCS", "3"),
        ],
    )
    signed_in = _make_profile(
        tmp_path / "Profiles" / "b.default",
        [
            (".google.com", "__Secure-1PSID", "psid"),
            (".google.com", "SID", "sid"),
        ],
    )
    corrupt = tmp_path / "Profiles" / "c.default"
    corrupt.mkdir(parents=True)
    (corrupt / "cookies.sqlite").write_bytes(b"not sqlite")

    monkeypatch.setattr(cookies_mod, "firefox_profiles", lambda: [logged_out, corrupt, signed_in])

    picked = cookies_mod.pick_profile()
    assert picked["profile"] == str(signed_in)
    assert picked["cookies"]["__Secure-1PSID"] == "psid"
