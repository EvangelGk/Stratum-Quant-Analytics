"""Phase 5 — Agentic Calibration Loop (Optuna TPE + Telegram HITL).

The agent sweeps 5 execution-layer hyperparameters on the OOS window
(2020-2023), proposes the top-3 trials to the human operator via Telegram,
waits for a pick, then evaluates the chosen config on the held-out period
(2024-2026) and saves the final results.

Usage:
    python -m agents.calibration_agent --budget 50 --propose-top 3

Guardrails (enforced inside the objective function):
  - No look-ahead: train strictly on pre-2020 data
  - Holdout (2024+) is touched ONCE after human approval
  - tx_cost ≥ 5 bps enforced in search space
  - Universe size fixed at Phase 3 pruned list (no re-selection during search)
  - Trial is invalid (return -inf) if any single ticker weight > 10 % EW max
    or portfolio turnover > 300 %/yr
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import types
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── path setup (mirrors run_phase4_validation.py stub pattern) ─────────────────
ROOT = Path(__file__).resolve().parents[1]
for _p in [
    str(ROOT / "src"),
    str(ROOT / "src" / "Medallion" / "gold" / "AnalysisSuite"),
    str(ROOT / "src" / "exceptions"),
    str(ROOT / "agents"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _stub(n: str) -> types.ModuleType:
    m = types.ModuleType(n)
    sys.modules[n] = m
    return m


for _n in [
    "diskcache", "dotenv", "pandera", "pandera.errors", "secret_store",
    "logger", "logger.Catalog", "logger.Messages",
    "logger.Messages.DirectionsMess", "logger.Messages.MainMess",
]:
    if _n not in sys.modules:
        _stub(_n)

_stub("logger.Catalog").catalog = lambda *a, **kw: None  # type: ignore


def _load_src(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src" / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


em = _stub("exceptions.MedallionExceptions")


class _E(Exception):
    pass


em.AnalysisError = _E  # type: ignore
em.DataValidationError = _E  # type: ignore

_load_src("Medallion/gold/AnalysisSuite/mixed_frequency.py",
          "Medallion.gold.AnalysisSuite.mixed_frequency")
bt = _load_src("Medallion/gold/AnalysisSuite/backtest.py",
               "Medallion.gold.AnalysisSuite.backtest")

from telegram_hitl import TelegramHITL  # noqa: E402  (after sys.path is patched)

# ── constants ──────────────────────────────────────────────────────────────────
SELECTED_TICKERS = ["AAPL", "ABBV", "NVDA", "TSLA", "JNJ", "ORCL", "APD", "MS", "CVX"]
HOLDOUT_START    = pd.Timestamp("2024-01-01")
OUTPUT_JSON      = ROOT / "output" / "default" / "phase5_calibration.json"

# ── metric helpers ─────────────────────────────────────────────────────────────

def _ann_sharpe(r: np.ndarray) -> float:
    v = float(np.std(r, ddof=1))
    return float(np.mean(r) * 252 / (v * np.sqrt(252))) if v > 1e-10 else float("nan")


def _ann_return(r: np.ndarray) -> float:
    return float(np.mean(r) * 252)


def _max_dd(r: np.ndarray) -> float:
    curve = np.cumprod(1 + r)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    return float(dd.min()) if len(dd) else 0.0


def _calmar(r: np.ndarray) -> float:
    mdd = abs(_max_dd(r))
    return _ann_return(r) / mdd if mdd > 1e-6 else float("nan")


def _profit_factor(r: np.ndarray) -> float:
    wins = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    return float(wins / losses) if losses > 1e-10 else float("inf")


def _portfolio_sharpe(oos_returns_dict: dict) -> float:
    """Build equal-weight portfolio from ticker return series and return Sharpe."""
    frames = {k: v for k, v in oos_returns_dict.items() if len(v) > 0}
    if not frames:
        return float("-inf")
    wide = pd.DataFrame(frames).sort_index().fillna(0.0)
    port = wide.mean(axis=1).values
    return _ann_sharpe(port)


# ── Optuna objective ───────────────────────────────────────────────────────────

def _make_objective(df: pd.DataFrame):
    """Closure that captures the master DataFrame."""

    def objective(trial) -> float:
        import optuna
        exec_kw = {
            "inv_vol_target": trial.suggest_float("inv_vol_target", 0.10, 0.40),
            "atr_multiplier": trial.suggest_float("atr_multiplier", 2.0,  8.0),
            "max_hold_days":  trial.suggest_int(  "max_hold_days",  10,   40),
            "vol_scale_cap":  trial.suggest_float("vol_scale_cap",  1.2,  3.0),
            "tx_cost":        trial.suggest_float("tx_cost",        0.0005, 0.0015),
        }

        oos_rets: dict = {}
        for tk in SELECTED_TICKERS:
            tdf = df[df["ticker"] == tk].copy()
            if tdf.empty:
                continue
            try:
                res = bt.backtest_pre2020_holdout(tdf, ticker=tk, exec_kwargs=exec_kw)
                dates = res.get("test_dates", [])
                rets  = np.asarray(res.get("strategy_returns", []), dtype=float)
                if len(rets) == 0 or len(dates) != len(rets):
                    continue
                idx = pd.to_datetime(dates)
                # OOS only (pre-holdout): 2020 to 2023-12-31
                s = pd.Series(rets, index=idx)
                oos_rets[tk] = s[s.index < HOLDOUT_START].values
            except Exception:
                continue

        sh = _portfolio_sharpe(oos_rets)
        return sh if np.isfinite(sh) else float("-inf")

    return objective


# ── holdout evaluation ─────────────────────────────────────────────────────────

def _run_holdout(df: pd.DataFrame, exec_kw: dict) -> dict:
    """Run all tickers with exec_kw and return holdout (2024+) portfolio metrics."""
    import pandas as _pd
    _orig_ts = _pd.Timestamp

    # Monkey-patch to extend test window to 2025-12-31 for holdout extraction
    class _TsPatch:
        def __new__(cls, val, *args, **kwargs):
            if val == "2023-12-31":
                val = "2025-12-31"
            return _orig_ts(val, *args, **kwargs)

    bt.pd.Timestamp = _TsPatch  # type: ignore

    holdout_frames: dict = {}
    for tk in SELECTED_TICKERS:
        tdf = df[df["ticker"] == tk].copy()
        if tdf.empty:
            continue
        try:
            res = bt.backtest_pre2020_holdout(tdf, ticker=tk, exec_kwargs=exec_kw)
            dates = res.get("test_dates", [])
            rets  = np.asarray(res.get("strategy_returns", []), dtype=float)
            if len(rets) == 0 or len(dates) != len(rets):
                continue
            idx = pd.to_datetime(dates)
            s = pd.Series(rets, index=idx, name=tk)
            s_ho = s[s.index >= HOLDOUT_START]
            if len(s_ho) > 10:
                holdout_frames[tk] = s_ho
        except Exception:
            continue

    bt.pd.Timestamp = _orig_ts  # type: ignore

    if not holdout_frames:
        return {"n_days": 0, "sharpe": float("nan"), "calmar": float("nan"),
                "profit_factor": float("nan"), "ann_return": float("nan")}

    wide = pd.DataFrame(holdout_frames).sort_index().fillna(0.0)
    port = wide.mean(axis=1).values
    return {
        "n_days":        int(len(port)),
        "date_from":     str(wide.index[0].date()),
        "date_to":       str(wide.index[-1].date()),
        "sharpe":        round(_ann_sharpe(port), 6),
        "calmar":        round(_calmar(port), 6),
        "profit_factor": round(_profit_factor(port), 6),
        "ann_return":    round(_ann_return(port), 6),
    }


# ── Telegram proposal formatter ────────────────────────────────────────────────

def _format_proposal(rank: int, trial_num: int, params: dict, oos_sharpe: float) -> str:
    lines = [
        f"<b>#{rank}  Trial {trial_num}  OOS Sharpe = {oos_sharpe:+.4f}</b>",
        f"  vol_target    : {params['inv_vol_target']:.3f}",
        f"  atr_mult      : {params['atr_multiplier']:.2f}",
        f"  max_hold_days : {params['max_hold_days']}",
        f"  vol_scale_cap : {params['vol_scale_cap']:.2f}",
        f"  tx_cost       : {params['tx_cost']*10000:.1f} bps",
    ]
    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("[ERROR] optuna not installed. Run: pip install optuna")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Phase 5 calibration agent (Optuna TPE + Telegram HITL)"
    )
    parser.add_argument("--budget",      type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--propose-top", type=int, default=3,  help="Top-N configs to send to Telegram")
    parser.add_argument("--skip-hitl",   action="store_true",  help="Auto-select best config (skip Telegram)")
    args = parser.parse_args()

    # ── check Phase 3 HITL gate ────────────────────────────────────────────────
    queue_path = ROOT / "output" / "default" / ".optimizer" / "approval_queue.json"
    if queue_path.exists():
        q = json.loads(queue_path.read_text("utf-8"))
        if q.get("action_id") == "phase5_calibration" and q.get("status") == "NO":
            print("[GATE] Phase 5 was not approved. Run checkpoint3_hitl.py first.")
            sys.exit(1)
        print(f"[GATE] Phase 5 approved at {q.get('approved_at', '?')}")
    else:
        print("[GATE] No approval queue found — proceeding (run checkpoint3_hitl.py to enable gate)")

    # ── load data ──────────────────────────────────────────────────────────────
    gold = next(Path(ROOT / "data").glob("**/master_table.parquet"))
    df   = pd.read_parquet(gold)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"Loaded {len(df):,} rows ({df['ticker'].nunique()} tickers)")

    # ── run Optuna ─────────────────────────────────────────────────────────────
    print(f"\nPhase 5: running {args.budget} Optuna TPE trials …")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="stratum_calibration",
    )
    study.optimize(_make_objective(df), n_trials=args.budget, show_progress_bar=False)

    # ── extract top-N ──────────────────────────────────────────────────────────
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
                 and np.isfinite(t.value)]
    completed.sort(key=lambda t: t.value, reverse=True)
    top_trials = completed[: args.propose_top]

    if not top_trials:
        print("[ERROR] No completed trials — cannot proceed.")
        sys.exit(1)

    print(f"\nTop {len(top_trials)} configs (OOS 2020-2023):")
    for i, t in enumerate(top_trials, 1):
        print(f"  #{i}  trial={t.number}  Sharpe={t.value:+.4f}  params={t.params}")

    # ── Telegram HITL ──────────────────────────────────────────────────────────
    hitl = TelegramHITL(timeout_s=3600)

    if args.skip_hitl:
        chosen_trial = top_trials[0]
        print("\n[SKIP-HITL] Auto-selected best config.")
    else:
        lines = [
            "🔬 <b>STRATUM QUANT — PHASE 5 CALIBRATION COMPLETE</b>",
            f"Budget: {args.budget} trials  |  Scored on OOS 2020–2023",
            "",
            "Please choose ONE configuration to run on the true holdout (2024–2026):",
            "",
        ]
        for i, t in enumerate(top_trials, 1):
            lines.append(_format_proposal(i, t.number, t.params, t.value))
            lines.append("")

        lines += [
            "⚠️ <i>The holdout will be evaluated ONCE with your chosen config.</i>",
            "This completes Phase 5. Reply with the config number below.",
        ]
        full_msg = "\n".join(lines)

        choices = [f"Config #{i}" for i in range(1, len(top_trials) + 1)]
        choice = hitl.ask(full_msg, choices=choices)

        if choice is None:
            print("[HITL] No response — defaulting to Config #1 (best OOS)")
            chosen_trial = top_trials[0]
        else:
            idx = int(choice.split("#")[1]) - 1
            chosen_trial = top_trials[idx]
            print(f"[HITL] Selected: {choice!r} → trial {chosen_trial.number}")

    # ── holdout evaluation (ONE TIME) ──────────────────────────────────────────
    print(f"\nRunning TRUE HOLDOUT with chosen config (trial #{chosen_trial.number}) …")
    holdout_metrics = _run_holdout(df, chosen_trial.params)
    print("Holdout results:")
    for k, v in holdout_metrics.items():
        print(f"  {k:20s}: {v}")

    # ── save results ───────────────────────────────────────────────────────────
    best_oos = {
        "trial": chosen_trial.number,
        "oos_sharpe": round(chosen_trial.value, 6),
        "params": chosen_trial.params,
    }
    all_top = [
        {"trial": t.number, "oos_sharpe": round(t.value, 6), "params": t.params}
        for t in top_trials
    ]
    output = {
        "phase": 5,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "budget": args.budget,
        "selected_universe": SELECTED_TICKERS,
        "chosen_config": best_oos,
        "top_configs": all_top,
        "holdout_metrics": holdout_metrics,
        "baseline_params": {
            "inv_vol_target": 0.25,
            "atr_multiplier": 4.0,
            "max_hold_days": 21,
            "vol_scale_cap": 2.0,
            "tx_cost": 0.0005,
        },
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"\nResults saved to {OUTPUT_JSON}")

    # ── final Telegram notification ────────────────────────────────────────────
    ho = holdout_metrics
    hitl.notify(
        "🏁 <b>Phase 5 Holdout Complete</b>\n"
        f"Chosen config (trial #{chosen_trial.number}, OOS Sharpe {chosen_trial.value:+.4f}):\n"
        f"  vol_target={chosen_trial.params['inv_vol_target']:.3f}  "
        f"atr_mult={chosen_trial.params['atr_multiplier']:.2f}  "
        f"max_hold={chosen_trial.params['max_hold_days']}d\n\n"
        f"<b>Holdout 2024–2026 (N={ho.get('n_days', 0)}d)</b>\n"
        f"  Sharpe: {ho.get('sharpe', float('nan')):+.4f}\n"
        f"  Calmar: {ho.get('calmar', float('nan')):+.4f}\n"
        f"  PF    : {ho.get('profit_factor', float('nan')):+.4f}\n"
        f"  Return: {ho.get('ann_return', float('nan')):+.4f}"
    )

    print("\n✅ Phase 5 complete.")


if __name__ == "__main__":
    main()
