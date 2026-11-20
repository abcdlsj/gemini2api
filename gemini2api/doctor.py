"""Sign in, look, and report — without generating anything.

`python -m gemini2api.doctor` prints what each account is, which models it may use, which of those
are subscription-only, and its quota windows. It never sends a prompt, so it spends no quota: the
only RPCs are the ones `init()` already makes.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from gemini_webapi.constants import AccountStatus
from loguru import logger

from .clients import make_client
from .config import Account, Settings, load


async def inspect(account: Account) -> dict[str, Any]:
    client = make_client(account)
    report: dict[str, Any] = {"account": account.id}
    try:
        await client.init(auto_refresh=False, verbose=False)
    except Exception as exc:
        report["error"] = str(exc)
        return report
    try:
        status = client.account_status
        report["status"] = status.name
        report["signed_in"] = status == AccountStatus.AVAILABLE
        if status != AccountStatus.AVAILABLE:
            report["detail"] = status.description
            return report
        models = client.list_models() or []
        report["models"] = [
            {
                "name": model.model_name,
                "display_name": model.display_name,
                "advanced_only": model.advanced_only,
                "available": model.is_available,
            }
            for model in models
        ]
        report["advanced_models"] = [m["name"] for m in report["models"] if m["advanced_only"]]
        report["quotas"] = client.quotas
        report["usage"] = client.usage_info
        return report
    finally:
        await client.close()


async def run(settings: Settings) -> int:
    reports = []
    for account in settings.accounts:
        reports.append(await inspect(account))

    print(json.dumps(reports, indent=2, ensure_ascii=False, default=str))

    failures = [r for r in reports if not r.get("signed_in")]
    if failures:
        for report in failures:
            reason = report.get("detail") or report.get("error") or "not signed in"
            logger.error(f"account {report['account']}: {reason}")
        return 1
    for report in reports:
        if not report.get("advanced_models"):
            logger.warning(
                f"account {report['account']}: no subscription-only model is offered. "
                "The cookies may belong to a free account, or the tier is not what you expect."
            )
    return 0


def main() -> int:
    try:
        settings = load()
    except Exception as exc:
        logger.error(f"configuration error: {exc}")
        return 2
    import os
    from pathlib import Path

    state = Path(settings.state_dir).expanduser()
    state.mkdir(parents=True, exist_ok=True)
    os.environ["GEMINI_COOKIE_PATH"] = str(state)
    return asyncio.run(run(settings))


if __name__ == "__main__":
    sys.exit(main())
