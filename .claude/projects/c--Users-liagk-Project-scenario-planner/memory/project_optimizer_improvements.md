---
name: Optimizer HITL and Telegram Improvements
description: Fixed ticker approval, added critical issues acceptance, exec risk assessment, pre-validation
type: project
---

Changes to src/optimizer.py (FullStackAuditOrchestrator + TelegramHITL):
1. Ticker expansion: removed `< 8` gate - now ALWAYS asks when recommendations exist
2. Critical issues: added `send_issues_approval()` method + Phase 3b in orchestrator
3. Execution risk assessment: populated R², look-ahead bias status, survivor bias note from governance JSON
4. Pre-validation: `_handle_code_fix()` now tests AI fixes with financial regression gate BEFORE showing to user

**Why:** User reported last 2 ticker additions didn't ask for acceptance; critical issues weren't getting accept/reject prompts; risk assessment showed N/A values.
**How to apply:** These changes are live. No config changes needed.
