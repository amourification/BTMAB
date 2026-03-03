# =============================================================================
#  charts/exporter.py — Chart & Data Export Module
#  PDF, PNG, and CSV export of all charts and trade plan data.
#  Summary page renderer and trade plan flattener live in exporter_utils.py.
#
#  Public interface:
#      export_png(chart_fn, plan, dir, suffix)  -> str path
#      export_pdf(plan, dir)                    -> str path
#      export_csv(plan, dir)                    -> str path
#      export_all(plan, dir, cfg)               -> dict {pdf, csv, success}
#      export_summary_png(plan, dir)            -> str path
# =============================================================================

import csv
import logging
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from charts.exporter_utils  import render_summary_page, flatten_trade_plan
from charts.cycle_chart     import render as render_cycle
from charts.murray_chart    import render as render_murray
from charts.phase_chart     import render as render_phase
from charts.risk_chart      import render as render_risk
from charts.gann_chart      import render as render_gann

logger = logging.getLogger("temporal_bot.charts.exporter")

BG_COLOR             = "#0d0f14"
DEFAULT_OUTPUT_DIR   = Path("/tmp/temporal_bot_exports")
DPI_SCREEN           = 120
DPI_PRINT            = 200
FIGSIZE_WIDE         = (16, 9)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _fname(symbol: str, suffix: str, ext: str) -> str:
    return f"{symbol.replace('/','').replace(':','')}_{suffix}_{_ts()}.{ext}"


# ── Public interfaces ─────────────────────────────────────────────────────────

def export_png(
    chart_fn,
    trade_plan: dict,
    output_dir: Path,
    suffix:     str,
    dpi:        int   = DPI_SCREEN,
    figsize:    tuple = FIGSIZE_WIDE,
) -> str:
    """
    Exports a single chart to PNG.

    Parameters
    ----------
    chart_fn   : callable — one of render_cycle / render_murray / etc.
    trade_plan : dict     — output of aggregator.run()
    output_dir : Path
    suffix     : str      — e.g. "cycle", "murray"
    dpi        : int
    figsize    : tuple

    Returns
    -------
    str — absolute path to the saved PNG (empty string on failure)
    """
    _ensure_dir(output_dir)
    path = output_dir / _fname(trade_plan.get("symbol", "UNK"), suffix, "png")
    try:
        fig = chart_fn(trade_plan, figsize=figsize)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close(fig)
        logger.info("PNG → %s", path)
        return str(path)
    except Exception as exc:
        logger.error("PNG export '%s' failed: %s", suffix, exc)
        return ""


def export_pdf(trade_plan: dict, output_dir: Path, dpi: int = DPI_PRINT) -> str:
    """
    Exports a 6-page PDF report:
      1 Summary · 2 Cycle · 3 Murray · 4 Phase/ACF · 5 Risk · 6 Gann

    Parameters
    ----------
    trade_plan : dict — output of aggregator.run()
    output_dir : Path
    dpi        : int

    Returns
    -------
    str — absolute path to PDF (empty string on failure)
    """
    _ensure_dir(output_dir)
    path = output_dir / _fname(trade_plan.get("symbol", "UNK"), "report", "pdf")
    pages = [
        ("summary", render_summary_page),
        ("cycle",   render_cycle),
        ("murray",  render_murray),
        ("phase",   render_phase),
        ("risk",    render_risk),
        ("gann",    render_gann),
    ]
    try:
        with PdfPages(path) as pdf:
            for label, fn in pages:
                try:
                    fig = fn(trade_plan)
                    pdf.savefig(fig, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR)
                    plt.close(fig)
                    logger.debug("PDF page: %s", label)
                except Exception as exc:
                    logger.warning("PDF page '%s' skipped: %s", label, exc)
            d = pdf.infodict()
            d["Title"]  = f"Temporal Analysis — {trade_plan.get('symbol','?')}"
            d["Author"] = "TemporalBot"
        logger.info("PDF → %s", path)
        return str(path)
    except Exception as exc:
        logger.error("PDF export failed: %s", exc)
        return ""


def export_csv(trade_plan: dict, output_dir: Path) -> str:
    """
    Exports flat trade plan metrics to CSV (one row).

    Parameters
    ----------
    trade_plan : dict — output of aggregator.run()
    output_dir : Path

    Returns
    -------
    str — absolute path to CSV (empty string on failure)
    """
    _ensure_dir(output_dir)
    path = output_dir / _fname(trade_plan.get("symbol", "UNK"), "trade_plan", "csv")
    try:
        flat = flatten_trade_plan(trade_plan)
        flat["export_timestamp"] = _ts()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(flat.keys()))
            w.writeheader(); w.writerow(flat)
        logger.info("CSV → %s", path)
        return str(path)
    except Exception as exc:
        logger.error("CSV export failed: %s", exc)
        return ""


def export_all(trade_plan: dict, output_dir: Path = None, cfg: dict = None) -> dict:
    """
    Exports PDF + CSV in one call.

    Parameters
    ----------
    trade_plan : dict — output of aggregator.run()
    output_dir : Path | None — defaults to DEFAULT_OUTPUT_DIR
    cfg        : dict | None — may contain "EXPORT_DIR", "EXPORT_DPI"

    Returns
    -------
    dict with keys: "pdf", "csv", "success"
    """
    cfg        = cfg or {}
    output_dir = Path(cfg.get("EXPORT_DIR", output_dir or DEFAULT_OUTPUT_DIR))
    dpi        = int(cfg.get("EXPORT_DPI", DPI_SCREEN))
    pdf_path   = export_pdf(trade_plan, output_dir, dpi=dpi)
    csv_path   = export_csv(trade_plan, output_dir)
    return {"pdf": pdf_path, "csv": csv_path, "success": bool(pdf_path and csv_path)}


def export_summary_png(trade_plan: dict, output_dir: Path = None) -> str:
    """
    Exports the summary page only as PNG — used by the Telegram bot
    for quick at-a-glance sharing without the full PDF.

    Returns
    -------
    str — absolute path to the PNG
    """
    return export_png(
        render_summary_page, trade_plan,
        Path(output_dir or DEFAULT_OUTPUT_DIR), "summary",
    )


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[1]))
    from charts.exporter_utils import make_smoke_test_plan

    plan    = make_smoke_test_plan()
    out_dir = P("/tmp/temporal_bot_exports")
    result  = export_all(plan, out_dir)
    summ    = export_summary_png(plan, out_dir)

    print("✅ Exporter smoke test")
    print(f"   PDF     : {result['pdf']}")
    print(f"   CSV     : {result['csv']}")
    print(f"   Summary : {summ}")
    print(f"   Success : {result['success']}")
