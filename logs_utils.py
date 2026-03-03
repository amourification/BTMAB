from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _safe_symbol(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in symbol or "UNKNOWN")


def _safe_interval(interval: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in interval or "UNK")


def write_analysis_log(trade_plan: Dict[str, Any]) -> Path | None:
    """
    Persist a full analysis snapshot to logs/<symbol>/<interval>/YYYY-MM-DD/ file.

    The filename includes a UTC timestamp so multiple runs per day are kept.
    """
    try:
        symbol = _safe_symbol(trade_plan.get("symbol", "UNKNOWN"))
        interval = _safe_interval(trade_plan.get("interval", trade_plan.get("tf", "UNK")))

        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        ts_str = now.strftime("%H%M%S")

        project_root = Path(__file__).resolve().parent
        base_dir = project_root / "logs" / symbol / interval / date_str
        base_dir.mkdir(parents=True, exist_ok=True)

        fname = f"{symbol}_{interval}_{date_str}_{ts_str}.json"
        fpath = base_dir / fname

        adv = trade_plan.get("advanced", {}) if isinstance(trade_plan, dict) else {}

        payload = {
            "symbol": symbol,
            "interval": interval,
            "generated_at_utc": now.isoformat(),
            "mode": trade_plan.get("mode", "classic") if isinstance(trade_plan, dict) else "classic",
            "vol_regime": adv.get("vol_regime"),
            "trend_regime": adv.get("trend_regime"),
            "uncertainty_score": adv.get("uncertainty_score"),
            "trade_plan": trade_plan,
        }
        with fpath.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)

        return fpath
    except Exception:
        # Logging failures must never break the GUI / analysis.
        return None

