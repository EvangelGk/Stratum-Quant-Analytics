"""
scheduler_batch.py — Pillar 4: Always-On Scheduler entry point.

Executed by .github/workflows/optimizer-scheduler.yml every 24 hours.
Runs in three phases:
  1. Full Medallion pipeline  (src/main.py)
  2. Grid-search backtest optimisation (src/optimizer.py → grid_search_backtest)
  3. Full-Stack Audit with Telegram HITL (src/optimizer.py → run_full_stack_audit)

Artifacts written to output/ are then committed back to the repo by the
workflow's "Commit and push" step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR   = REPO_ROOT / "src"
OUTPUT_DIR = REPO_ROOT / "output"


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess and raise on non-zero exit."""
    print(f"\n{'─' * 60}")
    print(f"[scheduler_batch] {label}")
    print(f"[scheduler_batch] cmd: {' '.join(cmd)}")
    print(f"{'─' * 60}\n")
    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(f"[scheduler_batch] ✗ {label} exited with code {result.returncode}",
              file=sys.stderr)
        sys.exit(result.returncode)
    print(f"[scheduler_batch] ✓ {label} completed successfully.")


def run_pipeline() -> None:
    """Phase 1 — execute the full Medallion data pipeline."""
    _run([sys.executable, str(SRC_DIR / "main.py")], "Full Medallion pipeline")


def run_grid_search() -> None:
    """
    Phase 2 — load the Gold master table and run the backtest grid search.

    Writes results to output/optimizer/grid_search_results.csv so the
    Streamlit app can display them without re-running the optimizer.
    """
    # The grid search needs price data.  We load from the Gold parquet when
    # available; otherwise skip gracefully so the pipeline commit still lands.
    master_path = REPO_ROOT / "data" / "gold" / "master_table.parquet"
    if not master_path.exists():
        print(
            "[scheduler_batch] Gold master_table.parquet not found — "
            "skipping grid search (pipeline may not have produced price data)."
        )
        return

    # Run grid search inline (same process) so it shares the venv cleanly.
    try:
        import sys as _sys
        if str(SRC_DIR) not in _sys.path:
            _sys.path.insert(0, str(SRC_DIR))
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))

        import pandas as pd
        from optimizer import grid_search_backtest  # type: ignore[import]

        df = pd.read_parquet(master_path)

        # Prefer 'adj_close', fall back to 'close'
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns:
            print(f"[scheduler_batch] Column '{price_col}' not in master table — skipping.")
            return

        # Use the most liquid ticker when multiple are present
        if "ticker" in df.columns:
            tickers = df["ticker"].dropna().unique().tolist()
            ticker = tickers[0] if tickers else None
            if ticker:
                df = df[df["ticker"] == ticker].copy()

        if "date" in df.columns:
            df = df.sort_values("date")
            prices = df.set_index("date")[price_col].dropna()
        else:
            prices = df[price_col].dropna()

        prices = pd.to_numeric(prices, errors="coerce").dropna()

        if len(prices) < 210:
            print(
                f"[scheduler_batch] Only {len(prices)} price rows — "
                "need ≥ 210 for 200-day trend SMA. Skipping grid search."
            )
            return

        print(f"[scheduler_batch] Running grid search on {len(prices)} price rows ...")
        output_path = str(OUTPUT_DIR / "optimizer" / "grid_search_results.csv")

        top5 = grid_search_backtest(
            prices=prices,
            rolling_windows=[10, 20, 50, 100],
            z_thresholds=[1.0, 1.5, 2.0, 2.5],
            volatility_filter=True,
            trend_filter=True,
            friction=0.0015,
            top_n=5,
            output_path=output_path,
        )

        # Also write a timestamped JSON summary for easy inspection
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "top5_by_sortino": top5,
        }
        summary_path = OUTPUT_DIR / "optimizer" / "grid_search_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )

        print(f"[scheduler_batch] Grid search complete. Top result: {top5[0] if top5 else 'N/A'}")
        print(f"[scheduler_batch] Results written to {output_path}")

    except Exception as exc:
        # Non-fatal: the pipeline artifacts are still committed even if the
        # grid search fails (e.g. missing dependency, corrupt parquet).
        print(f"[scheduler_batch] ✗ Grid search error (non-fatal): {exc}", file=sys.stderr)


def run_full_stack_audit_phase() -> None:
    """Phase 3 — Full-Stack Audit with Telegram HITL."""
    try:
        if str(SRC_DIR) not in sys.path:
            sys.path.insert(0, str(SRC_DIR))
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))

        from optimizer import run_full_stack_audit  # type: ignore[import]

        user_id = os.getenv("DATA_USER_ID", "default")
        timeout = int(os.getenv("FULL_STACK_AUDIT_TIMEOUT", "300"))

        print(f"[scheduler_batch] Running Full-Stack Audit (timeout={timeout}s) ...")
        result = run_full_stack_audit(
            project_root=REPO_ROOT,
            user_id=user_id,
            hitl_timeout=timeout,
        )
        decision = result.get("hitl_decision", "n/a")
        print(f"[scheduler_batch] ✓ Full-Stack Audit complete. HITL decision: {decision}")
    except Exception as exc:
        # Non-fatal: audit failure must not block artifact commits.
        print(f"[scheduler_batch] ✗ Full-Stack Audit error (non-fatal): {exc}", file=sys.stderr)


if __name__ == "__main__":
    print(f"[scheduler_batch] Starting at {datetime.now(timezone.utc).isoformat()}")
    run_pipeline()
    run_grid_search()
    run_full_stack_audit_phase()
    print(f"[scheduler_batch] All phases done at {datetime.now(timezone.utc).isoformat()}")
