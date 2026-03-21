# Llama 3.2 AI Integration for Quant Optimization

## Overview

The optimizer has been enhanced with **Llama 3.2:1b** AI agent integration via Ollama. This enables intelligent, autonomous problem diagnosis and solution recommendation with human approval gating for serious issues.

## Configuration

### Model Details
- **Model**: `llama3.2:1b` (installed locally via Ollama)
- **API Endpoint**: `http://localhost:11434/api/generate`
- **Temperature**: 0.3 (low randomness for consistency)
- **Timeout**: 120 seconds per analysis

### Optimizer Loop Settings
- **Max Iterations**: 8 (reduced from 10)
- **Approval Method**: YES/NO responses only (binary decisions)
- **Approval Gating**: Only for SERIOUS problems

## System Prompt

The Llama AI operates under this system prompt:

```
You are a Senior Quant Data Fixer AI Agent. Your role is to analyze financial data 
pipeline problems and recommend precise code fixes.

INPUT FORMAT:
- Task: [problem description from diagnostic]
- Rules: Only suggest fixes for SERIOUS problems (negative R², data gaps, 
         multicollinearity, drawdown >20%)
- Context: [relevant metrics and diagnostics]

OUTPUT FORMAT:
Generate a technical report with EXACTLY these sections:

---PROBLEM---
[Clear explanation of the issue and why it matters]

---ROOT_CAUSE---
[Technical root cause analysis]

---SOLUTION---
[Precise, actionable solution]

---CODE_CHANGES---
[Exact Python code changes needed with line references]

---APPROVAL_QUESTION---
Do you approve this solution? (YES/NO)

---
Keep all explanations concise and technical. No sugar-coating.
```

## Serious Problems Requiring Approval

The AI only requests human approval for these SERIOUS issues:

| Issue Type | Description | Threshold |
|---|---|---|
| `NEGATIVE_R2` | Out-of-sample R² is negative (weak predictive power) | r² < 0 |
| `DATA_GAPS` | Missing sources or failed data fetches | Any missing sources |
| `EXCESS_DRAWDOWN` | Maximum drawdown exceeds 20% | Impact > 0.20 |
| `MULTICOLLINEARITY` | Variance Inflation Factor exceeds healthy threshold | VIF > 5.0 |
| `REGIME_INSTABILITY` | Rolling elasticity coefficient of variation too high | CV > 200% |

Minor issues (outliers, scaling issues) skip approval and log as skipped adjustments.

## Approval Flow

### Interactive Mode (Terminal/SSH)
1. Llama analyzes the problem
2. AI recommendation is printed to terminal
3. System prompts: `"Approve this change? (YES/NO):"`
4. User enters: `YES` or `NO`
5. Decision is recorded to `output/.optimizer/approval_queue.json`

### Non-Interactive Mode (CI/CD/Automation)
1. Llama generates analysis
2. Approval request written to `output/.optimizer/approval_queue.json`
3. Process polls file for 120 seconds
4. Human owner edits file: set `"status": "YES"` or `"status": "NO"`
5. Process detects change and proceeds

## Modified Files

### `src/optimizer.py`

**New Class: `LlamaQuantAnalyzer`**
```python
class LlamaQuantAnalyzer:
    """AI-powered quantitative problem fixer using Llama 3.2 via Ollama."""
    
    def analyze_problem(problem_description: str, diagnostics: Dict) -> Dict:
        """Call Llama 3.2 to analyze problem and propose solution."""
        
    def _verify_connection() -> bool:
        """Verify Ollama service is running and model available."""
```

**Modified Class: `ApprovalGateway`**
- Changed response format from `"approved"/"rejected"` to `"YES"/"NO"`
- Updated `_prompt_interactive()` to accept `YES/NO` (case-insensitive)
- Updated `_poll_file_approval()` to watch for `YES/NO` status

**Modified Class: `AutomatedOptimizationLoop`**
- Constructor parameter: `max_iterations` default changed to **8** (was 10)
- New initialization: `self._llama = LlamaQuantAnalyzer(timeout_seconds=120)`
- New method: `_request_llama_approval_for_serious_issue()`
- Modified `run()` loop:
  - Negative R² → Uses Llama analysis before approval
  - Data gaps → Uses Llama analysis before approval
  - Outliers → Uses Llama analysis before approval

### `pyproject.toml`
- Added dependency: `"requests>=2.31.0,<3.0.0"`

## Usage

### Starting the Optimizer with Llama

```bash
# Ensure Ollama is running with the model
ollama serve

# In another terminal, run the optimizer
python src/optimizer.py --target-score 94.0 --max-iterations 8
```

### Example Output

```
[OPTIMIZER] ⚠️  APPROVAL REQUIRED BEFORE CODE MUTATION
  Action : [LLAMA AI] Serious problem detected: NEGATIVE_R2
  Details: {
    "iteration": 2,
    "issue_type": "NEGATIVE_R2",
    "current_score": 82.5,
    "inconsistencies": ["NEGATIVE_R2:governance_oos_r2=-0.1234"],
    "llama_analysis": "---PROBLEM---\nThe out-of-sample R² is negative..."
  }
========================================
  Approve this change? (YES/NO): YES
```

## Approval Queue File Format

Location: `output/<user_id>/.optimizer/approval_queue.json`

```json
{
  "action_id": "iter2_serious_NEGATIVE_R2",
  "description": "[LLAMA AI] Serious problem detected: NEGATIVE_R2",
  "details": {
    "iteration": 2,
    "issue_type": "NEGATIVE_R2",
    "current_score": 82.5,
    "inconsistencies": ["NEGATIVE_R2:governance_oos_r2=-0.1234"],
    "llama_analysis": "..."
  },
  "status": "pending",
  "requested_at": "2026-03-21T10:30:45.123456Z",
  "approved_at": null
}
```

Once approved, status becomes `"YES"`:

```json
{
  ...
  "status": "YES",
  "approved_at": "2026-03-21T10:30:52.456789Z"
}
```

## Verification Checklist

- [x] Ollama installed with `llama3.2:1b` model
- [x] Ollama API accessible at `localhost:11434`
- [x] `requests` library added to dependencies
- [x] Code is syntactically valid
- [x] Max iterations set to 8
- [x] Approval responses changed to YES/NO
- [x] Llama analyzer integrated into optimization loop
- [x] Only serious issues request approval

## Troubleshooting

### Connection Issues
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Verify model is loaded
ollama list | grep llama3.2:1b
```

### Timeout Issues
- Increase `LlamaQuantAnalyzer` timeout from 120 to 180 seconds
- Check Ollama process is not overloaded
- Reduce other system load

### Approval Stuck (Non-Interactive Mode)
- Check `output/.optimizer/approval_queue.json` exists
- Manually set status: `"YES"` or `"NO"`
- Save and ensure write permissions

## Architecture Notes

**Why Llama 3.2:1b?**
- Lightweight model (1.3 GB) runs on modest hardware
- Fast inference (~2-5 seconds per analysis)
- Sufficient reasoning for structured problem analysis
- Local execution (no API calls, data privacy maintained)

**Binary Approval (YES/NO)**
- Simpler than multi-option approval
- Faster and error-free human decisions
- Fits quant workflow: "do we apply this fix?"

**8 Iterations Cap**
- Reduced from 10 to reflect practical optimization feedback loop
- Each iteration now more intelligent (with AI guidance)
- Prevents infinite optimization spirals

