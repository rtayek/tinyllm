import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
PARENT = ROOT.parent
for candidate in (SRC, ROOT, PARENT):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
