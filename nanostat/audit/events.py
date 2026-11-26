"""Hash-chained audit event log (SPEC §10)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENESIS = "0" * 64


def _canon(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


class AuditLog:
    def __init__(self, deterministic: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self.deterministic = deterministic

    def add(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        prev = self.events[-1]["hash"] if self.events else GENESIS
        ev = {
            "seq": len(self.events),
            "ts": "1970-01-01T00:00:00+00:00" if self.deterministic else datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
            "prev": prev,
        }
        ev["hash"] = hashlib.sha256(_canon(ev).encode()).hexdigest()
        self.events.append(ev)
        return ev

    def to_jsonl(self) -> str:
        return "".join(_canon(e) + "\n" for e in self.events)

    def write(self, path: Path) -> None:
        path.write_text(self.to_jsonl(), encoding="utf-8")


def verify_chain(lines: list[str]) -> bool:
    prev = GENESIS
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        ev = json.loads(line)
        h = ev.pop("hash")
        if ev["prev"] != prev or ev["seq"] != i:
            return False
        if hashlib.sha256(_canon(ev).encode()).hexdigest() != h:
            return False
        prev = h
    return True
