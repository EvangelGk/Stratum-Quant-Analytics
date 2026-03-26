"""
scheduler_batch.py

This script is intended to be run by the GitHub Actions workflow to execute the optimizer and commit new results.
It should be idempotent and safe to run repeatedly.
"""
import subprocess
import sys
import os
from datetime import datetime

# Set up environment (if needed)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output')

# 1. Run the optimizer (replace with your actual optimizer command)
def run_optimizer():
    # Example: run the optimizer script (adjust as needed)
    result = subprocess.run([sys.executable, 'src/optimizer.py'], cwd=REPO_ROOT)
    if result.returncode != 0:
        print('Optimizer failed', file=sys.stderr)
        sys.exit(result.returncode)

# 2. Stage and commit new/changed artifacts
def commit_artifacts():
    subprocess.run(['git', 'config', '--global', 'user.email', 'github-actions[bot]@users.noreply.github.com'])
    subprocess.run(['git', 'config', '--global', 'user.name', 'github-actions[bot]'])
    subprocess.run(['git', 'add', 'output/'])
    commit_msg = f"Automated optimizer run: {datetime.utcnow().isoformat()}"
    subprocess.run(['git', 'commit', '-m', commit_msg, '--allow-empty'])
    subprocess.run(['git', 'push'])

if __name__ == '__main__':
    run_optimizer()
    commit_artifacts()
