"""Entrypoint: `python -m gemini2api` or `gemini2api`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn
from loguru import logger

from .config import load


def main() -> int:
    # A frozen binary has no `python -m`, so the helpers dispatch through argv.
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd in {"-h", "--help"}:
            print("usage: gemini2api [cookies ... | doctor]")
            return 0
        if cmd == "cookies":
            from .cookies import main as cookies_main

            return cookies_main(sys.argv[2:])
        if cmd == "doctor":
            from .doctor import main as doctor_main

            return doctor_main()

    try:
        settings = load()
    except Exception as exc:
        logger.error(f"configuration error: {exc}")
        return 2

    # Point the library's rotated-cookie cache at a stable directory before any client exists.
    state = Path(settings.state_dir).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    os.environ["GEMINI_COOKIE_PATH"] = str(state)

    logger.info(
        f"{len(settings.accounts)} account(s), chat_mode={settings.chat_mode}, "
        f"listening on {settings.host}:{settings.port}"
    )
    if not settings.api_key:
        logger.warning("no api_key configured: anyone who can reach this port can use the account")

    from .server import create_app

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
