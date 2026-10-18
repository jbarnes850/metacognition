"""JSON-lines files, optionally gzipped. No GPU dependencies, so analysis runs on CPU."""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path


def open_any(path: str | Path, mode: str = "rt"):
    path = str(path)
    return gzip.open(path, mode) if path.endswith(".gz") else open(path, mode)


def read_jsonl(path):
    with open_any(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)
