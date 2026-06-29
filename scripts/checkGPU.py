from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_gpu import *  # noqa: F403
from check_gpu import main


if __name__ == "__main__":
    main()
