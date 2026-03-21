#!/usr/bin/env python
"""
Approval Queue Manager - Simple CLI tool to approve/reject pending optimizer requests

USAGE:
    python respond_to_approval.py --approve    # Approve the pending request
    python respond_to_approval.py --reject     # Reject the pending request
    python respond_to_approval.py --status     # Show pending approval request
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


def get_approval_queue_path(user_id: str = "default") -> Path:
    """Get the approval queue file path."""
    return Path(f"output/{user_id}/.optimizer/approval_queue.json")


def read_approval_queue(user_id: str = "default") -> dict:
    """Read the approval queue file."""
    path = get_approval_queue_path(user_id)
    
    if not path.exists():
        return None
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Could not read approval queue: {e}")
        return None


def write_approval_queue(data: dict, user_id: str = "default") -> bool:
    """Write back the approval queue file."""
    path = get_approval_queue_path(user_id)
    
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"[ERROR] Could not write approval queue: {e}")
        return False


def show_status(user_id: str = "default"):
    """Display the current approval request status."""
    queue = read_approval_queue(user_id)
    
    if queue is None:
        print("[INFO] No approval queue file found.")
        print(f"       Expected location: {get_approval_queue_path(user_id)}")
        return
    
    status = queue.get("status", "unknown")
    
    if status == "pending":
        print("\n" + "="*70)
        print("PENDING APPROVAL REQUEST")
        print("="*70)
        print(f"\nAction ID: {queue.get('action_id', 'N/A')}")
        print(f"Description: {queue.get('description', 'N/A')}")
        print(f"Requested at: {queue.get('requested_at', 'N/A')}")
        
        details = queue.get("details", {})
        print(f"\nDetails:")
        print(f"  Iteration: {details.get('iteration', 'N/A')}")
        print(f"  Issue Type: {details.get('issue_type', 'N/A')}")
        print(f"  Current Score: {details.get('current_score', 'N/A')}/100")
        
        if isinstance(details.get('inconsistencies'), list):
            print(f"  Inconsistencies:")
            for inc in details['inconsistencies']:
                print(f"    - {inc}")
        
        if "llama_analysis" in details:
            print(f"\nAI Analysis (first 300 chars):")
            print(f"  {details['llama_analysis'][:300]}...")
        
        print("\n" + "="*70)
        print("RESPOND WITH:")
        print("  python respond_to_approval.py --approve   (to accept)")
        print("  python respond_to_approval.py --reject    (to decline)")
        print("="*70 + "\n")
        
    elif status == "YES":
        print(f"[APPROVED] Request was approved at {queue.get('approved_at', 'N/A')}")
    elif status == "NO":
        print(f"[REJECTED] Request was rejected at {queue.get('approved_at', 'N/A')}")
    else:
        print(f"[UNKNOWN] Status: {status}")


def approve(user_id: str = "default") -> bool:
    """Approve the pending request."""
    queue = read_approval_queue(user_id)
    
    if queue is None:
        print("[ERROR] No approval queue found.")
        return False
    
    if queue.get("status") != "pending":
        print(f"[INFO] Request status is already: {queue.get('status')}")
        return False
    
    queue["status"] = "YES"
    queue["approved_at"] = datetime.utcnow().isoformat() + "Z"
    
    if write_approval_queue(queue, user_id):
        print(f"[SUCCESS] Approved request: {queue.get('action_id', 'N/A')}")
        print(f"          Time: {queue['approved_at']}")
        print("\n[INFO] Scheduler will detect this change within 2 seconds and continue.")
        return True
    return False


def reject(user_id: str = "default") -> bool:
    """Reject the pending request."""
    queue = read_approval_queue(user_id)
    
    if queue is None:
        print("[ERROR] No approval queue found.")
        return False
    
    if queue.get("status") != "pending":
        print(f"[INFO] Request status is already: {queue.get('status')}")
        return False
    
    queue["status"] = "NO"
    queue["approved_at"] = datetime.utcnow().isoformat() + "Z"
    
    if write_approval_queue(queue, user_id):
        print(f"[SUCCESS] Rejected request: {queue.get('action_id', 'N/A')}")
        print(f"          Time: {queue['approved_at']}")
        print("\n[INFO] Scheduler will detect this change within 2 seconds and continue.")
        return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Manage optimizer approval requests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python respond_to_approval.py --status      # See what needs approval
  python respond_to_approval.py --approve     # Approve the request
  python respond_to_approval.py --reject      # Reject the request
        """
    )
    
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve the pending approval request",
    )
    parser.add_argument(
        "--reject",
        action="store_true",
        help="Reject the pending approval request",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show the current approval request status (default if no args)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for output/approvals",
    )
    
    args = parser.parse_args()
    
    # Default to status if no action specified
    if not (args.approve or args.reject or args.status):
        args.status = True
    
    if args.status:
        show_status(args.user_id)
    elif args.approve:
        approve(args.user_id)
    elif args.reject:
        reject(args.user_id)


if __name__ == "__main__":
    main()
