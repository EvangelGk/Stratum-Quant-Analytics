"""Traffic-light scoring utilities for the Scenario Planner UI.

Each helper returns a (color, label, description) tuple where color is one of
"green", "yellow", "red". Use ``badge_html()`` to render an inline HTML chip
that Streamlit can display via ``st.markdown(..., unsafe_allow_html=True)``.
"""
from __future__ import annotations

from typing import Optional

# ---------------------------------------------------------------------------
# Core primitives
# ---------------------------------------------------------------------------

_BADGE_CSS = (
    "display:inline-block;padding:2px 10px;border-radius:12px;"
    "font-weight:600;font-size:0.85rem;color:#fff;"
)
_COLOR_HEX = {"green": "#2e7d32", "yellow": "#f57c00", "red": "#c62828"}


def badge_html(label: str, color: str = "green", tooltip: str = "") -> str:
    """Return an inline HTML badge chip coloured by traffic-light status."""
    hex_color = _COLOR_HEX.get(color, _COLOR_HEX["red"])
    title_attr = f' title="{tooltip}"' if tooltip else ""
    return (
        f'<span style="{_BADGE_CSS}background:{hex_color}"{title_attr}>'
        f"{label}</span>"
    )


# ---------------------------------------------------------------------------
# Domain-specific scorers — each returns (color, short_label, business_desc)
# ---------------------------------------------------------------------------

def score_audit_status(status: str) -> tuple[str, str, str]:
    s = str(status).upper()
    if s == "PASS":
        return "green", "✅ All Clear", "All audit checks passed — system is production-ready."
    if s == "WARN":
        return "yellow", "⚠️ Review Needed", "Some checks flagged warnings — review before relying on output."
    return "red", "❌ Action Required", "One or more critical checks failed — pipeline output should not be used."


def score_decision_ready(ready: bool) -> tuple[str, str, str]:
    if ready:
        return "green", "✅ Decision Ready", "Analytical outputs are complete and validated."
    return "red", "❌ Not Ready", "Pipeline output is incomplete or blocked by quality gates."


def score_governance_gate(passed: bool, severity: str) -> tuple[str, str, str]:
    sev = str(severity).lower()
    if passed and sev in ("pass", "ok"):
        return "green", "✅ Gate Passed", "Governance checks are acceptable for mixed-frequency scenario analysis."
    if passed and sev == "warn":
        return "yellow", "⚠️ Gate Passed with Warnings", "Model passed with elevated risk indicators; use careful interpretation."
    if not passed and sev == "warn":
        return "yellow", "⚠️ Gate Soft-Fail", "Hard blockers were avoided, but non-R² governance risks require attention."
    return "red", "❌ Gate Failed", "Governance blocked outputs due to non-metric risk conditions (e.g., leakage/drift/model risk)."


def score_model_risk(score: Optional[float]) -> tuple[str, str, str]:
    if score is None:
        return "yellow", "Unknown", "Model risk score not available."
    s = float(score)
    if s <= 0.35:
        return "green", "Low Risk", f"Model risk score {s:.2f} — within safe operating range."
    if s <= 0.55:
        return "yellow", "Moderate Risk", f"Model risk score {s:.2f} — elevated but within tolerance."
    return "red", "High Risk", f"Model risk score {s:.2f} — exceeds acceptable ceiling."


def score_oos_r2(r2: Optional[float]) -> tuple[str, str, str]:
    if r2 is None:
        return "yellow", "No Data", "Out-of-sample R² not computed."
    v = float(r2)
    if v >= 0.05:
        return "green", "Good Fit", f"OOS R² = {v:.3f} — model explains meaningful variance."
    if v >= -0.50:
        return "yellow", "Weak Signal", f"OOS R² = {v:.3f} — negative values are common in noisy return forecasting; monitor with caution."
    return "red", "High Uncertainty", f"OOS R² = {v:.3f} — predictive signal is currently unstable and requires close review."


def score_null_pct(null_pct: Optional[float]) -> tuple[str, str, str]:
    if null_pct is None:
        return "yellow", "Unknown", "Null density not computed."
    v = float(null_pct)
    if v <= 10.0:
        return "green", "Dense", f"{v:.1f}% missing — data coverage is excellent."
    if v <= 25.0:
        return "yellow", "Sparse", f"{v:.1f}% missing — verify imputation strategy."
    return "red", "Very Sparse", f"{v:.1f}% missing — data gaps may compromise model reliability."


def score_source_coverage(pct: Optional[float]) -> tuple[str, str, str]:
    if pct is None:
        return "yellow", "Unknown", "Coverage not computed."
    v = float(pct)
    if v >= 80.0:
        return "green", "Good Coverage", f"{v:.1f}% row coverage from this source."
    if v >= 50.0:
        return "yellow", "Partial Coverage", f"{v:.1f}% row coverage — some rows lack this source."
    return "red", "Poor Coverage", f"{v:.1f}% row coverage — source integration may have failed."


def score_check_result(passed: bool, status: str) -> tuple[str, str, str]:
    s = str(status).lower()
    if passed and s == "pass":
        return "green", "✅ Pass", "Check passed without issues."
    if s == "warn" or (passed and s != "pass"):
        return "yellow", "⚠️ Warning", "Check raised warnings that merit review."
    return "red", "❌ Fail", "Check failed — requires investigation."
