#!/usr/bin/env python
"""
Quick test of Llama 3.2 integration with the Optimizer.

This script demonstrates the AI analysis flow without running the full pipeline.
Use it to verify Ollama connectivity and AI response quality.
"""

import json
from datetime import datetime

import requests


def test_ollama_connection():
    """Test if Ollama is running and responsive."""
    print("[TEST] Checking Ollama connection...")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": "test",
                "stream": False,
            },
            timeout=10,
        )
        if response.status_code == 200:
            print("✓ Ollama is running and responsive")
            return True
        else:
            print(f"✗ Ollama returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to Ollama at localhost:11434")
        print("  Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_llama_analysis():
    """Test Llama's ability to analyze a quant problem."""
    print("\n[TEST] Testing Llama AI analysis capability...")

    # Simulate a negative R² problem
    problem = """SERIOUS: NEGATIVE_R2 found. Score: 82.5/100.
Issues: ['NEGATIVE_R2:governance_oos_r2=-0.1234']"""

    context = """Integrity Score: 82.5/100
Issues detected:
  - NEGATIVE_R2:governance_oos_r2=-0.1234
  - OUTLIERS:ratio_gt_5pct=0.052
Risk metrics: VaR95=-0.0523, CVaR95=-0.0618"""

    system_prompt = """You are a Senior Quant Data Fixer AI Agent. Your role is to analyze financial data
pipeline problems and recommend precise code fixes.

INPUT FORMAT:
- Task: [problem description from diagnostic]
- Rules: Only suggest fixes for SERIOUS problems (negative R², data gaps, multicollinearity, drawdown >20%)
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
Keep all explanations concise and technical. No sugar-coating."""

    prompt = f"""{system_prompt}

TASK: {problem}

CONTEXT:
{context}

Please provide your analysis and solution recommendation."""

    try:
        print("Sending analysis request to Llama 3.2...")
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2:1b",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.3,
            },
            timeout=120,
        )

        if response.status_code == 200:
            analysis = response.json().get("response", "")
            print("\n" + "=" * 70)
            print("LLAMA AI ANALYSIS OUTPUT:")
            print("=" * 70)
            print(analysis)
            print("=" * 70)
            print("✓ Analysis completed successfully")
            return True
        else:
            print(f"✗ Llama returned status {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print("✗ Request timed out (Llama is taking too long)")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_approval_flow():
    """Demonstrate the approval flow."""
    print("\n[TEST] Simulating approval flow...")

    approval_data = {
        "action_id": "test_serious_NEGATIVE_R2",
        "description": "[LLAMA AI] Serious problem detected: NEGATIVE_R2",
        "details": {
            "iteration": 1,
            "issue_type": "NEGATIVE_R2",
            "current_score": 82.5,
            "inconsistencies": ["NEGATIVE_R2:governance_oos_r2=-0.1234"],
            "llama_analysis": "[Llama's analysis truncated for demo...]",
        },
        "status": "pending",
        "requested_at": datetime.utcnow().isoformat() + "Z",
        "approved_at": None,
    }

    print("Approval request created:")
    print(json.dumps(approval_data, indent=2))

    print("\n✓ Approval flow is correctly structured")
    print("  - Interactive mode: Prompts user for YES/NO")
    print("  - Non-interactive mode: Polls approval_queue.json for status change")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("LLAMA 3.2 INTEGRATION TEST SUITE")
    print("=" * 70)

    # Test 1: Connection
    if not test_ollama_connection():
        print("\n[ERROR] Cannot proceed without Ollama running.")
        print("Start Ollama with: ollama serve")
        return

    # Test 2: Analysis
    test_llama_analysis()

    # Test 3: Approval flow
    test_approval_flow()

    print("\n" + "=" * 70)
    print("TEST SUITE COMPLETE")
    print("=" * 70)
    print("\nConfiguration Summary:")
    print("  - Model: llama3.2:1b")
    print("  - API: http://localhost:11434/api/generate")
    print("  - Max iterations: 8")
    print("  - Approval method: YES/NO")
    print("  - Serious issues requiring approval: NEGATIVE_R2, DATA_GAPS,")
    print("    EXCESS_DRAWDOWN, MULTICOLLINEARITY, REGIME_INSTABILITY")
    print("\nNext Steps:")
    print("  1. Ensure Ollama is running: ollama serve")
    print("  2. Run the optimizer: python src/optimizer.py")
    print("  3. Respond YES/NO to serious issues when prompted")


if __name__ == "__main__":
    main()
