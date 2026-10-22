import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for d in ("processing", "estimation", "motion_model", "multipath"):
    sys.path.insert(0, str(ROOT / d))
