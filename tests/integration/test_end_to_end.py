"""End-to-end: >= 30 smoke tasks through the controller, audit bundle reproduction, CLI and API."""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from evaluation.metrics import evaluate_split
from nanostat.audit.bundle import build_bundle, verify_manifest
from nanostat.audit.events import verify_chain
from nanostat.controller import run_analysis
from nanostat.planner import LLMPlanner
from nanostat.schemas import Study


def test_smoke_tasks_end_to_end(smoke_data: Path):
    n = 0
    for sp in ("train", "dev", "test"):
        summ, outs = evaluate_split(smoke_data, sp)
        n += len(outs)
        assert all(o.operational_failure is None for o in outs), [o for o in outs if o.operational_failure]
    assert n >= 30


def test_rule_planner_is_valid_on_answerable_smoke_tasks(smoke_data: Path):
    summ, outs = evaluate_split(smoke_data, "train")
    assert summ.metrics["method_validity_rate"]["rate"] == 1.0
    assert summ.metrics["invalid_method_rate"]["k"] == 0


def test_controller_with_mocked_llm(paired_long):
    st = Study(roles={"y": "score", "group": "time", "pair_id": "id"}, group_order=["pre", "post"])
    outputs = iter(
        [
            "Sure! here you go: not json",
            '{"action":"execute","design":"independent_two_group","method":"welch_t",'
            '"columns":{"y":"score","group":"time"},"alternative":"greater"}',
            '{"action":"execute","design":"paired","method":"paired_t",'
            '"columns":{"y":"score","group":"time","pair_id":"id"},"alternative":"greater"}',
        ]
    )
    rec = run_analysis(
        "Did the score increase from pre to post?", st, paired_long, planner=LLMPlanner(lambda prompt: next(outputs))
    )
    assert rec.attempts == 3
    assert rec.result.method == "paired_t" and rec.result.method_status.value == "admissible"
    kinds = [json.loads(line)["kind"] for line in rec.audit.to_jsonl().splitlines()]
    assert kinds.count("plan_rejected") == 1 and kinds.count("verify") == 2


def test_bundle_reproduces_in_fresh_process(tmp_path: Path, two_group):
    st = Study(roles={"y": "sbp", "group": "arm"}, group_order=["control", "treatment"])
    rec = run_analysis("Does treatment reduce sbp compared with control?", st, two_group)
    csv_bytes = two_group.to_csv(index=False).encode()
    z = build_bundle(rec, csv_bytes, tmp_path / "b.zip")
    assert verify_manifest(z)
    work = tmp_path / "second_machine"
    with zipfile.ZipFile(z) as zf:
        zf.extractall(work)
        assert verify_chain(zf.read("audit_events.jsonl").decode().splitlines())
    r = subprocess.run([sys.executable, "-I", "analysis.py"], cwd=work, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert abs(out["p_value"] - rec.result.p_value) < 1e-12
    adm = pd.read_csv(work / "admissible_range.csv")
    assert rec.result.method in set(adm["method"])


def test_audit_chain_detects_tampering(two_group):
    st = Study(roles={"y": "sbp", "group": "arm"}, group_order=["control", "treatment"])
    rec = run_analysis("Is there a difference in sbp between control and treatment?", st, two_group)
    lines = rec.audit.to_jsonl().splitlines()
    assert verify_chain(lines)
    ev = json.loads(lines[-1])
    ev["payload"]["p_value"] = 0.0001
    assert not verify_chain(lines[:-1] + [json.dumps(ev)])


def test_cli_analyze_and_bundle(tmp_path: Path, two_group, monkeypatch):
    from nanostat.cli import app

    monkeypatch.setenv("NANOSTAT_RUNS_DIR", str(tmp_path / "runs"))
    csv = tmp_path / "d.csv"
    two_group.to_csv(csv, index=False)
    study = tmp_path / "s.json"
    study.write_text(json.dumps({"roles": {"y": "sbp", "group": "arm"}, "group_order": ["control", "treatment"]}))
    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "analyze",
            "-q",
            "Does treatment reduce sbp compared with control?",
            "--csv",
            str(csv),
            "--study",
            str(study),
            "--rules-only",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output)
    assert payload["result"]["method"] == "welch_t" and payload["result"]["alternative"] == "less"
    res2 = runner.invoke(app, ["bundle", "--run", payload["run_id"], "--out", str(tmp_path / "b.zip")])
    assert res2.exit_code == 0 and verify_manifest(tmp_path / "b.zip")
    res3 = runner.invoke(app, ["analyze", "-q", "x?", "--csv", str(csv), "--adapter", "a"])
    assert res3.exit_code == 2
    res4 = runner.invoke(app, ["bundle", "--run", "../../etc", "--out", str(tmp_path / "x.zip")])
    assert res4.exit_code == 2


def test_api(tmp_path: Path, two_group, monkeypatch):
    from nanostat.api import app

    monkeypatch.setenv("NANOSTAT_RUNS_DIR", str(tmp_path / "runs"))
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    body = {
        "question": "Is there a difference in sbp between control and treatment?",
        "csv": two_group.to_csv(index=False),
        "study": {"roles": {"y": "sbp", "group": "arm"}, "group_order": ["control", "treatment"]},
    }
    r = c.post("/v1/analyze", json=body)
    assert r.status_code == 200, r.text
    rid = r.json()["run_id"]
    assert c.get(f"/v1/runs/{rid}").json()["result"]["p_value"] is not None
    b = c.get(f"/v1/runs/{rid}/bundle")
    assert b.status_code == 200 and b.content[:2] == b"PK"
    assert c.get("/v1/runs/..%2F..%2Fsecret").status_code == 404
    assert c.post("/v1/analyze", json={**body, "server_path": "/etc/passwd"}).status_code == 422
