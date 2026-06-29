from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_bottleneck_profiler import *  # noqa: F403
from data_bottleneck_profiler import main


if __name__ == "__main__":
    main()
