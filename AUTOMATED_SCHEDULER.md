# Automated Daily Optimizer with Scheduled Execution

## Overview

The optimizer now runs **automatically every day** without requiring you to call it manually. When serious problems are found, it writes approval requests to a JSON file and **waits for your approval** before making changes.

## How It Works

### The Flow

```
┌─────────────────────────────────────────────────────────────┐
│ [DAILY 2:00 AM] Windows Task Scheduler runs               │
│ run_optimizer_daily.bat                                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ Python scheduler.py --once                                  │
│ Runs optimizer with scheduled=True                          │
├─────────────────────────────────────────────────────────────┤
│ Iteration 1: Diagnoses problems...                         │
│   └─ SERIOUS ISSUE FOUND!                                   │
│      Writes to: output/default/.optimizer/approval_queue.json
│      Status: "pending"                                      │
│      Waits for your approval...                             │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ YOU (at your leisure)│
         │ Edit approval file  │
         │ Set status: "YES"   │
         │ Submit form         │
         └──────────┬──────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│ Scheduler detects change                                    │
│ Continues optimization with approved fix                    │
│ Next iterations proceed automatically                       │
└─────────────────────────────────────────────────────────────┘
```

## Installation (Windows Task Scheduler)

### Step 1: Open Task Scheduler

```
Windows Key + R → taskschd.msc → Enter
```

### Step 2: Create New Task

1. **Right-click** on "Task Scheduler Library" → **Create Basic Task**
2. Fill in details:
   - **Name**: `Daily Scenario Planner Optimizer`
   - **Description**: `Automated daily optimization with file-based approvals`
   - **Check**: "Run whether user is logged in or not"
   - **Check**: "Run with highest privileges" (if needed for network access)

### Step 3: Set Trigger

1. Click **Triggers** tab → **New**
2. Choose: **Daily**
3. Set time: **02:00 AM** (or your preferred time)
4. Click OK

### Step 4: Set Action

1. Click **Actions** tab → **New**
2. **Action**: "Start a program"
3. **Program/script**: 
   ```
   C:\Users\liagk\Project\scenario-planner\run_optimizer_daily.bat
   ```
4. **Start in**: (leave empty)
5. Click OK

### Step 5: Set Conditions (Optional but Recommended)

1. Click **Conditions** tab
2. **Power**: 
   - ✓ "Start the task only if the computer is on AC power" (uncheck if laptop)
   - ✓ "Stop if the computer switches to battery power" (uncheck if laptop)
3. Click OK

### Step 6: Save

Click **OK** to save the task. Windows will ask for your password.

---

## How to Approve/Reject Changes

### When Running

The scheduler will write approval requests like this:

**File location**: `output/default/.optimizer/approval_queue.json`

```json
{
  "action_id": "iter1_serious_NEGATIVE_R2",
  "description": "[LLAMA AI] Serious problem detected: NEGATIVE_R2",
  "details": {
    "iteration": 1,
    "issue_type": "NEGATIVE_R2",
    "current_score": 82.5,
    "llama_analysis": "---PROBLEM---\nThe out-of-sample R² is negative..."
  },
  "status": "pending",
  "requested_at": "2026-03-21T02:15:30.123456Z",
  "approved_at": null
}
```

### To Approve (or Reject)

#### Option A: Edit File Directly (Recommended)
1. Open: `C:\Users\liagk\Project\scenario-planner\output\default\.optimizer\approval_queue.json`
2. Change line: `"status": "pending"` to `"status": "YES"` or `"status": "NO"`
3. **Save file** (Ctrl+S)
4. Scheduler detects change immediately (polls every 2 seconds)
5. Optimization continues with approved fix

#### Option B: Terminal/PowerShell
```powershell
# Approve
(Get-Content 'output/default/.optimizer/approval_queue.json') -replace '"status": "pending"', '"status": "YES"' | Set-Content 'output/default/.optimizer/approval_queue.json'

# Reject
(Get-Content 'output/default/.optimizer/approval_queue.json') -replace '"status": "pending"', '"status": "NO"' | Set-Content 'output/default/.optimizer/approval_queue.json'
```

---

## Running Manually (Not Scheduled)

If you want to run the optimizer immediately (without waiting for scheduled time):

### Interactive Mode (Manual with Terminal Prompts)
```bash
cd C:\Users\liagk\Project\scenario-planner
python src/optimizer.py --target-score 94.0
```
When issues are found, the system prompts you in terminal:
```
[OPTIMIZER] ⚠️  APPROVAL REQUIRED BEFORE CODE MUTATION
  Approve this change? (YES/NO): YES
```

### Non-Interactive Mode (File-Based Approvals)
```bash
python src/scheduler.py --once --user-id default
```
Creates: `output/default/.optimizer/approval_queue.json` and polls for your response.

### Continuous Loop (Runs Repeatedly)
```bash
python src/scheduler.py --interval 24  # Every 24 hours
python src/scheduler.py --interval 12  # Every 12 hours
```

---

## Monitoring

### View Logs

Scheduler logs are written to:
```
logs/scheduler_YYYYMMDD.log
output/default/.scheduler/scheduler_YYYYMMDD.log
```

Each entry includes:
- Timestamp
- Event type (optimizer_start, optimizer_complete, optimizer_error, etc.)
- Details (score, iteration count, errors)

**Example log entry**:
```json
{"timestamp": "2026-03-21 02:15:30", "event": "optimizer_start", "details": {"target_score": 94.0}}
{"timestamp": "2026-03-21 02:45:15", "event": "optimizer_complete", "details": {"score": 95.2, "status": "optimized", "iterations": 3}}
```

### View Optimization Report

After each run, the full report is saved to:
```
output/default/optimizer_report.json
```

This includes:
- Final integrity score
- All inconsistencies found
- Complete Quant evaluation
- Loop history (each iteration's actions)

---

## Configuration

### Change Daily Schedule

Edit `run_optimizer_daily.bat`:

```batch
REM Change this line to adjust max iterations:
"%PYTHON_EXE%" "%SCHEDULER_SCRIPT%" --once --user-id "%USER_ID%"

REM Or run the continuous scheduler (uncomment to use):
REM "%PYTHON_EXE%" "%SCHEDULER_SCRIPT%" --interval 24 --user-id "%USER_ID%"
```

### Change Scheduled Time

1. Open Task Scheduler
2. Find "Daily Scenario Planner Optimizer"
3. Right-click → **Edit**
4. Click **Triggers** tab
5. Double-click the trigger
6. Change **Start time** to your preferred hour

### Run at Startup Instead of Daily

1. In Task Scheduler, edit the task
2. Click **Triggers** → **Edit**
3. Change from **Daily** to **At startup**
4. Click OK

---

## Troubleshooting

### Task Scheduler Says "Last Run Failed"

Check the log:
```
logs/scheduler.log
```

**Common issues:**
- Ollama service not running
  - Fix: Start Ollama manually: `ollama serve`
- Python path incorrect
  - Fix: Update path in `run_optimizer_daily.bat`
- Permissions issue
  - Fix: Run `taskschd.msc` as Administrator

### File Change Not Detected

The scheduler polls every 2 seconds. If approval takes longer:

**Increase timeout** in `src/optimizer.py`:
```python
def _ensure_approval_gateway(self) -> ApprovalGateway:
    if self._approval is None:
        self._approval = ApprovalGateway(
            base_output_path=self.output_dir.parent,
            timeout_seconds=300,  # 5 minutes instead of 120
            force_non_interactive=self.scheduled,
        )
    return self._approval
```

### Too Many Approvals to Handle

Modify the scheduler to request approval only for CRITICAL issues:

In `src/optimizer.py`, find `_request_llama_approval_for_serious_issue()` and add additional filters:
```python
# Only request approval if score drops significantly
if diag["score"] < 50:  # Only for severe problems
    is_serious = True
else:
    is_serious = False
```

---

## Summary

| Mode | How to Start | Approvals | Best For |
|------|---|---|---|
| **Interactive** | `python src/optimizer.py` | Terminal YES/NO | Manual runs, testing |
| **Scheduled (File-based)** | Windows Task Scheduler | Edit JSON file | Production, daily automation |
| **Once (Non-interactive)** | `python src/scheduler.py --once` | Poll file for 2 min | One-time automated run |
| **Loop (Non-interactive)** | `python src/scheduler.py --interval 24` | Poll file for 2 min | Continuous background process |

---

## Next Steps

1. ✓ IP address changed to `127.0.0.1:11434`
2. ✓ Scheduler created and tested
3. **Install task in Windows Task Scheduler** (see steps above)
4. Monitor logs daily
5. Respond to approvals as needed

