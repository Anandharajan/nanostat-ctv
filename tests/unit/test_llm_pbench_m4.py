"""Local-LLM integration (mock OpenAI-compatible server), LLM eval harness, P-Bench runner/scoring,
policy pre-review worksheet, TRL reward adapter, H1 protocol parsing and regression verification."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nanostat.llm import LLMError, OpenAICompatClient
from nanostat.profiler import profile
from nanostat.protocol import build_protocol
from nanostat.schemas import Plan, Study
from nanostat.verifier import verify

REPLY = {"text": '{"action":"abstain","reason_code":"no_admissible_method"}'}


class _H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "mock-1.5b"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers["Content-Length"])
        req = json.loads(self.rfile.read(n))
        assert req["model"] == "mock-1.5b" and req["messages"][-1]["role"] == "user"
        body = json.dumps(
            {
                "choices": [{"message": {"content": REPLY["text"]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), _H)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/inference/v1"
    srv.shutdown()


def test_client_roundtrip_and_local_only(server):
    c = OpenAICompatClient("mock-1.5b", server)
    assert c.list_models() == ["mock-1.5b"]
    assert "abstain" in c.generate("hi") and c.stats.completion_tokens == 5
    with pytest.raises(LLMError):
        OpenAICompatClient("m", "https://api.example.com/v1")
    dead = OpenAICompatClient("m", "http://127.0.0.1:9/v1", timeout=2)
    assert dead.generate("x").startswith("[LLM_ERROR]")  # operational failure, not a crash


def test_llm_eval_harness_is_resumable(server, smoke_data: Path, tmp_path: Path):
    from evaluation.llm_eval import run_llm_eval

    traces = tmp_path / "t.jsonl"

    def gen(trial: int):
        return OpenAICompatClient("mock-1.5b", server).generate

    r1 = run_llm_eval(smoke_data, "train", gen, "B1:mock", traces, trials=1, max_steps=1, limit=5)
    n1 = len(traces.read_text().splitlines())
    r2 = run_llm_eval(smoke_data, "train", gen, "B1:mock", traces, trials=1, max_steps=1, limit=5)
    assert n1 == len(traces.read_text().splitlines()) == 5 and r1["n_rows"] == r2["n_rows"] == 5
    assert r1["first_attempt_json_valid_rate"] == 1.0
    assert r1["total_completion_tokens"] == 5 * 5 and r1["total_llm_calls"] == 5  # mock reports 5 tokens per call
    row = json.loads(traces.read_text().splitlines()[0])
    assert row["prompt_hash"] and row["raw_outputs"]


def test_pbench_scoring_from_traces(tmp_path: Path, monkeypatch):
    import benchmark.pbench.runner as runner
    from benchmark.pbench.score import Answer

    answers = [Answer("cat/a", 0.01, "reject", "hard", "cat"), Answer("cat/b", 0.5, "fail_to_reject", "easy", "cat")]
    monkeypatch.setattr(runner, "load_answers", lambda: answers)
    tr = tmp_path / "pb.jsonl"
    rows = [
        {"label": "L", "task": "cat/a", "trial": i, "status": "ok", "p_value": 0.011, "decision": "reject"}
        for i in range(3)
    ]
    rows += [{"label": "L", "task": "cat/b", "trial": 0, "status": "abstained", "p_value": None, "decision": None}]
    tr.write_text("\n".join(json.dumps(r) for r in rows))
    rep = runner.score_pbench_traces(tr, "L")
    assert rep["composite_hard_strict_pass1"] == 100.0 and rep["difficulty:easy"]["strict_pass1"] == 0.0


def test_pbench_preflight_on_local_data(tmp_path: Path):
    from benchmark.pbench.loader import local_dir
    from benchmark.pbench.preflight import preflight

    if not (local_dir() / "splits.json").exists():
        pytest.skip("P-Bench not downloaded (run `nanostat pbench --download`)")
    rep = preflight(tmp_path)
    assert rep["tasks"] == 425 and rep["category_ceiling"]["tasks"] >= 416


def test_policy_prereview_never_marks_reviewed(tmp_path: Path):
    from nanostat.policy import load_policies
    from nanostat.policy_review import run_review, write_worksheet

    reply = json.dumps(
        {
            "admissible": [{"method": "welch_t", "verdict": "DISAGREE", "reason": "x"}],
            "invalid": [],
            "missing_methods": ["brunner_munzel"],
            "overall": "NEEDS_CHANGES",
            "comment": "",
        }
    )
    reviews = run_review(lambda p: reply)
    out = tmp_path / "w.md"
    write_worksheet(reviews, "mock", out)
    text = out.read_text(encoding="utf-8")
    assert "NOT a statistician sign-off" in text and "brunner_munzel" in text
    assert all(not c.confirmatory_ok for c in load_policies().values())


def test_trl_reward_adapter(smoke_data: Path):
    from training.constraint import ScenarioConstraint
    from training.rl_rewards import TaskStore, make_reward
    from training.sft_data import gold_plan

    store = TaskStore(smoke_data)
    tids = [t for t, k in store.keys.items() if k.abstain_target is None][:6]
    gold = [gold_plan(store.keys[t], store.tasks[t], store.df(t)).model_dump_json() for t in tids]
    f = make_reward(store, "ctv", ScenarioConstraint(pwr_ref={}))
    r = f(prompts=[""] * 6, completions=gold, task_id=tids)
    assert min(r) >= 0.85
    assert make_reward(store, "fisher")(prompts=[""], completions=["garbage"], task_id=tids[:1]) == [0.0]
    conv = [[{"role": "assistant", "content": gold[0]}]]
    assert make_reward(store, "ctv")(prompts=[""], completions=conv, task_id=tids[:1])[0] >= 0.85


def test_protocol_reads_h1_line():
    q = "Research Question:\nDoes higher stage predict worse survival?\nH0: no difference\nH1: survival differs by stage"
    assert build_protocol(q, Study()).alternative.value == "two-sided"
    assert build_protocol("Q\nH1: the treatment reduces depression", Study()).alternative.value == "less"


def test_regression_verification_paths():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=80), "y": rng.normal(size=80)})
    df["yb"] = (df.y > 0).astype(int)
    q = "Is x associated with y?"
    p = Plan.model_validate({"action": "execute", "design": "regression", "method": "ols_coef"})
    assert any(e.startswith("missing_model") for e in verify(p, df, profile(df), Study(), q).errors)
    p = Plan.model_validate(
        {
            "action": "execute",
            "design": "regression",
            "method": "ols_coef",
            "model": {"outcome": "nope", "exposure": ["x"]},
        }
    )
    assert any(e.startswith("unknown_column:model") for e in verify(p, df, profile(df), Study(), q).errors)
    p = Plan.model_validate(
        {"action": "execute", "design": "regression", "method": "tobit", "model": {"outcome": "yb", "exposure": ["x"]}}
    )
    assert not verify(p, df, profile(df), Study(), q).ok
    p = Plan.model_validate(
        {
            "action": "execute",
            "design": "regression",
            "method": "logistic",
            "model": {"outcome": "yb", "exposure": ["x"]},
        }
    )
    rep = verify(p, df, profile(df), Study(), q)
    assert rep.ok and rep.hypothesis == "coef_zero"
