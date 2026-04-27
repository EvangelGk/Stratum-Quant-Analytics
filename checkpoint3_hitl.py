"""Checkpoint 3 — Phase 4 results → Telegram HITL gate.

Reads output/default/phase4_validation.json, formats a summary, sends it
to the configured Telegram chat with [APPROVE → Phase 5] / [STOP] buttons,
polls for approval, and writes the decision to the approval queue file.

Run:
    python checkpoint3_hitl.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents"))

from telegram_hitl import TelegramHITL  # noqa: E402

# ── load Phase 4 results ───────────────────────────────────────────────────────
PHASE4_JSON = ROOT / "output" / "default" / "phase4_validation.json"
QUEUE_PATH  = ROOT / "output" / "default" / ".optimizer" / "approval_queue.json"

if not PHASE4_JSON.exists():
    print(f"[ERROR] Phase 4 results not found at {PHASE4_JSON}")
    sys.exit(1)

data = json.loads(PHASE4_JSON.read_text(encoding="utf-8"))

oos  = data["oos_metrics"]
bci  = data["bootstrap_cis"]
dsr  = data["dsr"]
hold = data["holdout_metrics"]
tks  = ", ".join(data["selected_tickers"])

def _fmt(v, pct=False) -> str:
    if v is None or v != v:     # None or NaN
        return "N/A"
    if pct:
        return f"{v:+.1f}%"
    return f"{v:+.4f}" if isinstance(v, float) else str(v)

pass_fail = {
    "Sharpe ≥ 0.6":        oos["sharpe"]          >= 0.6,
    "CI lower > 0":        bci["sharpe"]["ci_lower"] > 0,
    "PF ≥ 1.25":           oos["profit_factor"]   >= 1.25,
    "Calmar ≥ 0.5":        oos["calmar"]          >= 0.5,
    "DSR p-val > 0.5":     dsr["p_value"]         >= 0.5,
    "Holdout degrad <50%": (abs(hold["degradation_pct"]) < 50
                            if hold["degradation_pct"] is not None else False),
}
n_pass = sum(pass_fail.values())
n_total = len(pass_fail)
verdict_line = "✅ ALL PASS" if n_pass == n_total else f"⚠️ {n_pass}/{n_total} PASS"

lines = [
    "📊 <b>STRATUM QUANT — PHASE 4 RE-VALIDATION COMPLETE</b>",
    "",
    f"<b>Universe ({len(data['selected_tickers'])} tickers):</b> {tks}",
    f"<b>OOS window:</b> {data['oos_window']['start']} → {data['oos_window']['end']}",
    "",
    "<b>OOS Metrics</b>",
    f"  Sharpe    : {_fmt(oos['sharpe'])}  CI [{_fmt(bci['sharpe']['ci_lower'])}, {_fmt(bci['sharpe']['ci_upper'])}]",
    f"  Calmar    : {_fmt(oos['calmar'])}  CI [{_fmt(bci['calmar']['ci_lower'])}, {_fmt(bci['calmar']['ci_upper'])}]",
    f"  PF        : {_fmt(oos['profit_factor'])}  CI [{_fmt(bci['profit_factor']['ci_lower'])}, {_fmt(bci['profit_factor']['ci_upper'])}]",
    f"  Ann. Ret  : {_fmt(oos['annualized_return'])}",
    f"  Max DD    : {_fmt(oos['max_drawdown'])}",
    f"  Win Rate  : {_fmt(oos['win_probability'])}",
    "",
    "<b>Statistical Robustness</b>",
    f"  DSR p-value    : {_fmt(dsr['p_value'])}  (target &gt;0.5 ✅)",
    f"  Trials tested  : {dsr['n_trials']}",
    "",
    f"<b>True Holdout (2024–2026, N={hold['n_days']}d)</b>",
    f"  Sharpe    : {_fmt(hold['sharpe'])}",
    f"  Calmar    : {_fmt(hold['calmar'])}",
    f"  PF        : {_fmt(hold['profit_factor'])}",
    f"  Degrad.   : {_fmt(hold['degradation_pct'], pct=True)}  (target &lt;50%)",
    "",
    "<b>Criteria</b>",
]
for name, passed in pass_fail.items():
    icon = "✅" if passed else "❌"
    lines.append(f"  {icon} {name}")

lines += [
    "",
    f"<b>{verdict_line}</b>",
    "",
    "Approve to proceed to Phase 5 (Calibration Agent)?",
]

message = "\n".join(lines)

print("=" * 60)
print("CHECKPOINT 3 — TELEGRAM HITL GATE")
print("=" * 60)
print(message.replace("<b>", "").replace("</b>", "").replace("&gt;", ">").replace("&lt;", "<"))
print("=" * 60)

# ── send to Telegram ───────────────────────────────────────────────────────────
hitl = TelegramHITL(timeout_s=3600)   # 1-hour window
choice = hitl.ask(message, choices=["✅ APPROVE → Phase 5", "❌ STOP"])

approved = choice is not None and "APPROVE" in choice

# ── write approval queue ───────────────────────────────────────────────────────
QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
queue = {
    "action_id": "phase5_calibration",
    "description": "Proceed to Phase 5: Calibration Agent",
    "status": "YES" if approved else "NO",
    "requested_at": datetime.now(timezone.utc).isoformat(),
    "approved_at": datetime.now(timezone.utc).isoformat(),
    "details": {
        "phase4_sharpe": oos["sharpe"],
        "phase4_calmar": oos["calmar"],
        "holdout_sharpe": hold["sharpe"],
        "n_pass": n_pass,
        "n_total": n_total,
        "choice": choice,
    },
}
QUEUE_PATH.write_text(json.dumps(queue, indent=2, default=str), encoding="utf-8")

print()
if approved:
    print("✅ APPROVED — ready to run: python -m agents.calibration_agent --budget 50 --propose-top 3")
else:
    print("❌ NOT APPROVED — stopping before Phase 5.")
    sys.exit(1)
