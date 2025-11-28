import sys
from pathlib import Path

# Ensure project root (tiny/), src/, and parent are on sys.path for imports
ROOT = Path(__file__).resolve().parent.parent  # .../agents/tiny
SRC = ROOT / "src"
PARENT = ROOT.parent  # .../agents
for candidate in (SRC, ROOT, PARENT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
