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

        def sig(k):  # method AND specification requirements (spec twins keep the method name)
            req = sorted(
                json.dumps([m.method, {x: y for x, y in m.config.items() if x != "model"}], sort_keys=True)
                for m in k.admissible
            )
            return (k.abstain_target, k.preferred, k.alternative, tuple(req))

        gold_a, gold_b = sig(a), sig(b)
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


def test_group_exclusive_splits(smoke_data: Path):
    from benchmark.split import HELD_OUT_DOMAINS, HELD_OUT_SCENARIOS

    keys = load_keys(smoke_data / "keys" / "keys.jsonl")
    for k in keys.values():
        if k.split != "train":
            continue
        assert k.twin_edit not in HELD_OUT_EDITS
        assert (k.dgp.family, k.dgp.perturbation) not in HELD_OUT_SCENARIOS
        dom = int(k.lineage.template_group.split("_d")[-1].split("_")[0])
        assert dom not in HELD_OUT_DOMAINS
        if k.twin_id:
            assert keys[k.twin_id].split == "train"


def test_null_labels_are_hypothesis_specific():
    from benchmark.factory.dgp import null_labels

    uv = null_labels("two_group", "unequal_variance", "null")
    assert uv["mean_equal"] and not uv["distribution_equal"]
    lev = null_labels("correlation", "outliers", "null")
    assert lev["rho_zero"] and not lev["independence"]
    sk = null_labels("paired", "skewed", "null")
    assert sk["mean_diff_zero"] and not sk["symmetric_about_zero"]
    assert not any(null_labels("two_group", "none", "medium").values())


def test_estimand_change_twin_flips_method(tmp_path: Path):
    from benchmark.factory import render
    from benchmark.factory.dgp import Scenario, simulate
    from benchmark.factory.grid import build_task
    from benchmark.factory.twins import base_scenario, make_twin

    sc0 = Scenario("two_group", "medium", "medium", "none", "two-sided", "hint", 77, 0, 3, {"variant": 0})
    sc = base_scenario("estimand_change", sc0, "changed")
    sim = simulate(sc, render.names_for(sc))
    b1 = build_task(sc, sim, "sg", "a", "estimand_change", "changed")
    sc2, sim2 = make_twin("estimand_change", sc, sim, "changed")
    b2 = build_task(sc2, sim2, "sg", "b", "estimand_change", "changed")
    assert not isinstance(b1, str) and not isinstance(b2, str)
    assert b1.key.protocol.estimand_family == "mean" and b2.key.protocol.estimand_family == "typical"
    assert {a.method for a in b1.key.admissible}.isdisjoint({a.method for a in b2.key.admissible})
    assert b1.key.hypothesis == "mean_equal" and b2.key.hypothesis in ("distribution_equal", "trimmed_mean_equal")
