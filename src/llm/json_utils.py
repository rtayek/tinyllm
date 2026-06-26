from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any, *, ensure_ascii: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=ensure_ascii) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
