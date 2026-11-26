"""Local run storage shared by the CLI and the API."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from nanostat.audit.bundle import build_bundle
from nanostat.controller import RunRecord
from nanostat.security import InputError

RUN_ID_CHARS = set("0123456789abcdef")


def runs_dir() -> Path:
    return Path(os.environ.get("NANOSTAT_RUNS_DIR", "nanostat_runs"))


def run_id_for(question: str, study_json: str, data: bytes) -> str:
    h = hashlib.sha256()
    for part in (question.encode(), study_json.encode(), data):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:16]


def check_run_id(run_id: str) -> str:
    if len(run_id) != 16 or not set(run_id) <= RUN_ID_CHARS:
        raise InputError("invalid run id")
    return run_id


def save_run(rec: RunRecord, data: bytes) -> tuple[str, Path]:
    rid = run_id_for(rec.question, rec.study.model_dump_json(), data)
    d = runs_dir() / rid
    d.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "run_id": rid,
        "question": rec.question,
        "plan": rec.plan.model_dump(mode="json") if rec.plan else None,
        "result": rec.result.model_dump(mode="json"),
    }
    (d / "run.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    build_bundle(rec, data, d / "bundle.zip")
    return rid, d


def load_run(run_id: str) -> dict[str, Any]:
    p = runs_dir() / check_run_id(run_id) / "run.json"
    if not p.exists():
        raise FileNotFoundError(run_id)
    return json.loads(p.read_text(encoding="utf-8"))


def bundle_path(run_id: str) -> Path:
    p = runs_dir() / check_run_id(run_id) / "bundle.zip"
    if not p.exists():
        raise FileNotFoundError(run_id)
    return p
