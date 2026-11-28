"""Prompt v3 and field-level re-plan feedback (after the v2 B1 baseline, docs/LLM_B1_BASELINE.md)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nanostat.controller import Budget, run_analysis
from nanostat.planner import LLMPlanner, plan_errors
from nanostat.prompting import PROMPT_VERSION, render_prompt
from nanostat.schemas import ROLE_NAMES, Check, ReasonCode, Study

# The exact first output of Qwen2.5-Coder-1.5B on dev task 1 with prompt v2 (results/llm_b1/).
V2_FAILURE = (
    '```json\n{"action": "clarify", "design": "correlation", "method": "pearson", "estimand": "typical", '
    '"columns": {"fertiliser": "fertiliser", "yield_t": "yield_t"}, '
    '"checks": [{"check": "pairing", "verdict": "violated"}], "rationale": "no pairing info"}\n```'
)


def _task():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"score": rng.normal(0, 1, 40), "arm": ["a"] * 20 + ["b"] * 20})
    study = Study(roles={"y": "score", "group": "arm"})
    return "Does the treatment change the typical score compared with control?", study, df


def test_prompt_v3_enumerates_allowed_values():
    q, study, df = _task()
    from nanostat.profiler import profile

    p = render_prompt(q, study, profile(df))
    assert PROMPT_VERSION == "typed-plan-v3"
    for v in [*ROLE_NAMES, *(c.value for c in Check), *(r.value for r in ReasonCode)]:
        assert v in p
    assert 'Omit "estimand"' in p and "Copy it from Study.roles" in p and "no code fences" in p
    assert '"estimand": "<optional>"' not in p  # the v2 placeholder that invited invented values


def test_plan_errors_name_fields_and_allowed_values():
    errs = plan_errors(V2_FAILURE)
    assert any(e.startswith("estimand:") and "omit estimand" in e for e in errs)
    assert any(e.startswith("columns:") and "copy columns from Study.roles" in e for e in errs)
    assert plan_errors("no json here") == [
        "no JSON object found: output exactly one JSON object, with no code fences or prose"
    ]
    ok = '{"action":"execute","design":"independent_two_group","method":"welch_t","columns":{"y":"s","group":"g"}}'
    assert plan_errors(ok) == []
    clar = plan_errors('{"action":"clarify","design":"paired"}')
    assert any("reason_code" in e and "missing_pair_id" in e for e in clar)


def test_controller_feeds_field_level_errors_back():
    q, study, df = _task()
    prompts: list[str] = []
    outputs = iter([V2_FAILURE, '{"action":"abstain","reason_code":"no_admissible_method"}'])

    def gen(prompt: str) -> str:
        prompts.append(prompt)
        return next(outputs)

    run_analysis(q, study, df, planner=LLMPlanner(gen), budget=Budget(max_steps=2))
    assert len(prompts) == 2
    second = prompts[1]
    assert "Verifier feedback on your previous plan" in second
    assert "columns: unknown column roles" in second and "copy columns from Study.roles" in second
    assert "output was not a valid typed plan JSON" not in second


def test_score_traces_reports_recorded_prompt_version(tmp_path):
    import json
    from pathlib import Path

    from evaluation.llm_eval import score_traces

    src = Path(__file__).resolve().parents[2] / "results" / "llm_b1" / "qwen25coder15b_dev_steps1.jsonl"
    rows = src.read_text(encoding="utf-8").splitlines()[:3]
    f = tmp_path / "t.jsonl"
    f.write_text("\n".join(rows) + "\n", encoding="utf-8")
    r = score_traces(f)
    assert r["prompt_version"] == json.loads(rows[0])["prompt_version"] == "typed-plan-v2"


def test_sft_records_are_conversational_and_train_only(tmp_path):
    """SFT must use the chat format the evaluator uses, and gold plans must come from the train split only."""
    import json

    from training.train_gpu import sft_records

    f = tmp_path / "sft.jsonl"
    f.write_text(
        json.dumps({"task_id": "t1", "prompt": "P", "completion": '{"action":"abstain"}'}) + "\n", encoding="utf-8"
    )
    (r,) = sft_records(str(f))
    assert r == {
        "prompt": [{"role": "user", "content": "P"}],
        "completion": [{"role": "assistant", "content": '{"action":"abstain"}'}],
    }
    import inspect

    from training.sft_data import build_sft

    assert inspect.signature(build_sft).parameters["split"].default == "train"
