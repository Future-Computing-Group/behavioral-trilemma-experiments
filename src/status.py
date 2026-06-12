"""Per-task status emitter (L63; EXPERIMENT-PLAN §11.E-B.5).

Every long-running loop writes a STATUS.json after EVERY task —
unconditional, not gated on expected duration. Atomic tmp+rename so a
mid-write crash never leaves a torn file.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib


def write_status(path: pathlib.Path, **fields) -> None:
    """Atomically write ``fields`` + a ``last_update`` ISO timestamp to
    ``path`` (json.dump to ``<path>.tmp``, then ``os.replace``)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(fields)
    payload["last_update"] = _dt.datetime.now().isoformat(timespec="seconds")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
