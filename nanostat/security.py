"""Input limits, safe paths and a network guard for evaluation mode."""

from __future__ import annotations

import io
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLS = 200


class InputError(ValueError):
    pass


def safe_join(base: Path, rel: str) -> Path:
    """Resolve `rel` under `base`, rejecting absolute paths and traversal.

    Backslashes are treated as separators on every OS, so `..\\x` is rejected on Linux too."""
    if not rel or Path(rel).is_absolute() or rel.startswith(("/", "\\")) or ":" in rel:
        raise InputError(f"path must be relative: {rel!r}")
    rel = rel.replace("\\", "/")
    if ".." in rel.split("/"):
        raise InputError(f"path escapes base directory: {rel!r}")
    base_r = base.resolve()
    p = (base_r / rel).resolve()
    if base_r != p and base_r not in p.parents:
        raise InputError(f"path escapes base directory: {rel!r}")
    return p


def read_csv_limited(data: bytes) -> pd.DataFrame:
    if len(data) > MAX_BYTES:
        raise InputError(f"CSV exceeds {MAX_BYTES // (1024 * 1024)} MB")
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as e:  # noqa: BLE001 - surface parse errors as input errors
        raise InputError(f"could not parse CSV: {e}") from e
    if len(df) > MAX_ROWS:
        raise InputError(f"CSV has {len(df)} rows (limit {MAX_ROWS})")
    if df.shape[1] > MAX_COLS:
        raise InputError(f"CSV has {df.shape[1]} columns (limit {MAX_COLS})")
    return df


class NetworkBlockedError(RuntimeError):
    pass


@contextmanager
def no_network() -> Iterator[None]:
    """Block outbound socket connections (evaluation mode)."""
    orig_connect = socket.socket.connect
    orig_create = socket.create_connection

    def _blocked(*_a: Any, **_k: Any) -> Any:
        raise NetworkBlockedError("network access is blocked in evaluation mode")

    socket.socket.connect = _blocked  # type: ignore[method-assign]
    socket.create_connection = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.create_connection = orig_create  # type: ignore[assignment]
