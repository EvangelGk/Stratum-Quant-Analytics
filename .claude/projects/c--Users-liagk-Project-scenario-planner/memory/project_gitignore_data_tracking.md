---
name: Data and Output Tracking in Git
description: data/users/default/ and output/default/ are now committed to git so Streamlit Cloud has pipeline artifacts without needing a re-run
type: project
---

data/users/default/ and output/default/ are now tracked in git (committed in f52512c).

**Why:** Streamlit Cloud resets the ephemeral filesystem on each deployment. Without committing pipeline artifacts, Gold/Output files disappear after every push. The last 3 commits caused a redeploy that wiped the cloud filesystem.

**How to apply:** After each pipeline run locally, commit updated files in data/users/default/ and output/default/ (excluding *.gz backup archives which are gitignored). Push will auto-update Streamlit Cloud.

The .gitignore was restructured:
- `data/*` + `!data/users/` + `data/users/*` + `!data/users/default/` (nested negation pattern)
- `output/*` + `!output/default/` + `output/default/*.gz` (exclude compressed backups)
