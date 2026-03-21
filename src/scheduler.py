#!/usr/bin/env python
"""
Automated Daily Optimizer Scheduler
Runs the optimization loop periodically and polls for approvals via file.

Usage:
    python src/scheduler.py --interval=24  # Run every 24 hours
    python src/scheduler.py --interval=12  # Run every 12 hours  
    python src/scheduler.py --once         # Run once (useful for cron/task scheduler)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from optimizer import AutomatedOptimizationLoop


def log_scheduled_event(user_id: str, event: str, details: dict = None):
    """Log scheduler events to a timestamped log file."""
    log_dir = Path(f"output/{user_id}/.scheduler")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "event": event,
        "details": details or {},
    }
    
    log_file = log_dir / f"scheduler_{datetime.now().strftime('%Y%m%d')}.log"
    try:
        if log_file.exists():
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        else:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"[SCHEDULER] Warning: Could not write log: {e}")


def run_optimizer_once(target_score: float = 94.0, user_id: str = "default"):
    """Execute the optimizer in non-interactive mode (file-based approvals only)."""
    print(f"\n[SCHEDULER] Starting optimization loop at {datetime.now()}")
    print(f"[SCHEDULER] Target score: {target_score}")
    print(f"[SCHEDULER] User: {user_id}")
    print(f"[SCHEDULER] Mode: NON-INTERACTIVE (approvals via file polling)")
    
    log_scheduled_event(user_id, "optimizer_start", {"target_score": target_score})
    
    try:
        optimizer = AutomatedOptimizationLoop(
            target_score=target_score,
            max_iterations=8,
            user_id=user_id,
            scheduled=True,  # Force non-interactive, file-based approvals
        )
        
        report = optimizer.run()
        
        # Log completion
        log_scheduled_event(user_id, "optimizer_complete", {
            "score": report.get("raw_integrity_score"),
            "status": report.get("status"),
            "iterations": report.get("iterations_used"),
        })
        
        print(f"[SCHEDULER] Optimization complete at {datetime.now()}")
        print(f"[SCHEDULER] Final score: {report.get('raw_integrity_score')}/100")
        print(f"[SCHEDULER] Status: {report.get('status')}")
        print(f"[SCHEDULER] Iterations used: {report.get('iterations_used')}/{report.get('max_iterations')}")
        
        return report
        
    except Exception as e:
        log_scheduled_event(user_id, "optimizer_error", {"error": str(e)})
        print(f"[SCHEDULER] Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return None


def scheduler_loop(interval_hours: int = 24, user_id: str = "default"):
    """Run optimizer repeatedly at specified intervals."""
    print(f"\n[SCHEDULER] Starting daily optimizer scheduler")
    print(f"[SCHEDULER] Interval: {interval_hours} hour(s)")
    print(f"[SCHEDULER] User: {user_id}")
    print(f"[SCHEDULER] Approval method: FILE-BASED POLLING")
    print(f"[SCHEDULER] To approve/reject, edit: output/{user_id}/.optimizer/approval_queue.json")
    print(f"[SCHEDULER] Change 'status': 'pending' to 'status': 'YES' or 'NO'")
    
    log_scheduled_event(user_id, "scheduler_start", {"interval_hours": interval_hours})
    
    next_run = datetime.now()
    
    try:
        while True:
            now = datetime.now()
            
            if now >= next_run:
                # Run the optimizer
                run_optimizer_once(target_score=94.0, user_id=user_id)
                
                # Schedule next run
                next_run = now + timedelta(hours=interval_hours)
                print(f"\n[SCHEDULER] Next run scheduled for: {next_run}")
                log_scheduled_event(user_id, "next_run_scheduled", {
                    "next_run": next_run.isoformat()
                })
            
            # Sleep for 1 minute before checking again
            time.sleep(60)
            
    except KeyboardInterrupt:
        print(f"\n[SCHEDULER] Scheduler stopped by user at {datetime.now()}")
        log_scheduled_event(user_id, "scheduler_stop", {"reason": "user_interrupt"})
    except Exception as e:
        print(f"\n[SCHEDULER] Fatal error: {e}")
        log_scheduled_event(user_id, "scheduler_error", {"error": str(e)})
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Automated daily optimizer scheduler with file-based approval polling."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=24,
        help="Run optimizer every N hours (default: 24 = daily)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run optimizer once and exit (useful for cron/task scheduler)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default="default",
        help="User ID for output/logs",
    )
    parser.add_argument(
        "--target-score",
        type=float,
        default=94.0,
        help="Target optimization score",
    )
    
    args = parser.parse_args()
    
    if args.once:
        # Run once and exit
        run_optimizer_once(target_score=args.target_score, user_id=args.user_id)
    else:
        # Run in loop
        scheduler_loop(interval_hours=args.interval, user_id=args.user_id)


if __name__ == "__main__":
    main()
