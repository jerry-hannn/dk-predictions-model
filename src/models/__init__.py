"""Quantitative NFL & NBA game and player-prop models."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import joblib


def atomic_joblib_dump(obj: Any, target: Path) -> Path:
    """
    Serialize ``obj`` to ``target`` atomically (write temp file + os.replace) so concurrent
    readers never observe a partially written artifact while a retrain is in progress.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    os.close(fd)
    try:
        joblib.dump(obj, tmp_name)
        os.replace(tmp_name, target)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return target
