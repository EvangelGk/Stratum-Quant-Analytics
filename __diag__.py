import json
from pathlib import Path

p = Path("output/default/audit_report.json")
d = json.loads(p.read_text())
checks = d.get("checks", {})

print("=== CHECKS ===")
for name, val in checks.items():
    t = type(val).__name__
    is_dict = isinstance(val, dict)
    keys = list(val.keys())[:4] if is_dict else "N/A"
    print(f"  {name}: type={t}, is_dict={is_dict}, sample_keys={keys}")

print("=== COMPLETENESS LOGIC ===")
has_status = isinstance(d.get("status"), str) and bool(str(d.get("status", "")).strip())
has_decision = isinstance(d.get("decision_ready"), bool)
core = ["integration", "density", "statistics", "continuity", "survivorship", "outputs", "thresholds", "governance"]
present = sum(1 for n in core if isinstance(checks.get(n), dict))
has_core = present >= 4

rc = d.get("row_count")
cc = d.get("column_count")
counters_valid = True
if rc is not None:
    counters_valid = counters_valid and isinstance(rc, int) and rc > 0
if cc is not None:
    counters_valid = counters_valid and isinstance(cc, int) and cc > 0

print(f"has_status: {has_status}")
print(f"has_decision: {has_decision}")
print(f"present_core_checks: {present} >= 4: {has_core}")
print(f"row_count: {rc} ({type(rc).__name__})")
print(f"column_count: {cc} ({type(cc).__name__})")
print(f"counters_valid: {counters_valid}")
print(f"FINAL: {has_status and has_decision and has_core and counters_valid}")

print()
print("=== TOP-LEVEL KEYS ===")
for k, v in d.items():
    if k != "checks":
        print(f"  {k}: {type(v).__name__} = {repr(v)[:80]}")
