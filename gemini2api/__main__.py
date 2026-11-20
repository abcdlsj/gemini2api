"""Entrypoint: `python -m gemini2api` or `gemini2api`."""

from __future__ import annotations

import sys

import uvicorn
from loguru import logger

from .config import load


def main() -> int:
    try:
        settings = load()
    except Exception as exc:
        logger.error(f"configuration error: {exc}")
        return 2

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
