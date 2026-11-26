"""Factory, twins and split tests (SPEC §6.4, §7.3, §13)."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.factory.grid import run_factory
from benchmark.keys import load_keys
from benchmark.split import HELD_OUT_EDITS
from nanostat.schemas import TaskRecord


def _tasks(d: Path) -> dict[str, tuple[str, TaskRecord]]:
    out = {}
    for sp in ("train", "dev", "test"):
        for line in (d / "tasks" / f"{sp}.jsonl").read_text(encoding="utf-8").splitlines():
            t = TaskRecord.model_validate_json(line)
            out[t.task_id] = (sp, t)
    return out


def test_twins_share_split_and_change_gold(smoke_data: Path):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    tasks = _tasks(smoke_data)
    pairs = [(k, keys[k.twin_id]) for k in keys.values() if k.twin_id and k.task_id < k.twin_id]
    assert len(pairs) >= 10
    for a, b in pairs:
        assert tasks[a.task_id][0] == tasks[b.task_id][0] == a.split == b.split
        assert a.twin_edit == b.twin_edit
        gold_a = (a.abstain_target, a.preferred, a.alternative, tuple(sorted(m.method for m in a.admissible)))
        gold_b = (b.abstain_target, b.preferred, b.alternative, tuple(sorted(m.method for m in b.admissible)))
        assert gold_a != gold_b, a.twin_edit


def test_wording_fixed_twins_have_identical_question(smoke_data: Path):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    tasks = _tasks(smoke_data)
    fixed = [k for k in keys.values() if k.twin_wording == "fixed"]
    assert fixed
    for k in fixed:
        assert tasks[k.task_id][1].question == tasks[k.twin_id][1].question  # type: ignore[index]


def test_held_out_edits_only_in_test(smoke_data: Path):
    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    for k in keys.values():
        if k.twin_edit in HELD_OUT_EDITS:
            assert k.split == "test"
    assert len(HELD_OUT_EDITS) >= 2


def test_factory_is_deterministic(tmp_path: Path, smoke_data: Path):
    run_factory("smoke", tmp_path, seed=20261126)
    a = (tmp_path / "manifests" / "data_manifest.sha256").read_text()
    b = (smoke_data / "manifests" / "data_manifest.sha256").read_text()
    assert a == b


def test_task_files_contain_no_answer_fields(smoke_data: Path):
    for sp in ("train", "dev", "test"):
        text = (smoke_data / "tasks" / f"{sp}.jsonl").read_text(encoding="utf-8")
        for bad in ("h0_true", "admissible", "reference_decision", "discordance", "twin_id", "policy_cell"):
            assert bad not in text


def test_report_and_histogram(smoke_data: Path):
    rep = json.loads((smoke_data / "manifests" / "factory_report.json").read_text())
    assert rep["tasks"] >= 50 and rep["twin_pairs"] >= 10
    hist = (smoke_data / "manifests" / "discordance_hist.csv").read_text().splitlines()
    assert len(hist) == 11 and sum(int(r.split(",")[2]) for r in hist[1:]) == rep["tasks"]
    assert set(rep["per_wording"]) == {"fixed", "changed"}
