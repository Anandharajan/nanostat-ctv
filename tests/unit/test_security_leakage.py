"""Security and leakage tests (SPEC §13)."""

from __future__ import annotations

import ast
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from nanostat.security import InputError, NetworkBlockedError, no_network, read_csv_limited, safe_join

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "nanostat"
R_WORKER = PKG / "parity" / "r_backend.py"


def _calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                base = f.value.id if isinstance(f.value, ast.Name) else ""
                names.add(f"{base}.{f.attr}")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            names |= {f"import:{m}" for m in mods}
    return names


@pytest.mark.parametrize("path", sorted(p for p in PKG.rglob("*.py")), ids=lambda p: p.name)
def test_no_eval_exec_shell_outside_r_worker(path: Path):
    calls = _calls(path)
    assert not ({"eval", "exec", "compile", "__import__"} & calls), path
    assert "os.system" not in calls and "os.popen" not in calls
    if path != R_WORKER:
        assert "import:subprocess" not in calls, f"subprocess used outside the R worker: {path}"


def test_path_traversal_rejected(tmp_path: Path):
    (tmp_path / "ok.csv").write_text("a\n1\n")
    assert safe_join(tmp_path, "ok.csv").name == "ok.csv"
    for bad in ("../etc/passwd", "..\\..\\x.csv", "/etc/passwd", "C:\\Windows\\win.ini", "sub/../../x"):
        with pytest.raises(InputError):
            safe_join(tmp_path, bad)


def test_csv_limits():
    with pytest.raises(InputError):
        read_csv_limited(("c" + ",c".join(str(i) for i in range(250)) + "\n").encode())
    with pytest.raises(InputError):
        read_csv_limited(b"x" * (21 * 1024 * 1024))


def test_network_blocked_in_eval_mode():
    with no_network():
        with pytest.raises(NetworkBlockedError):
            socket.create_connection(("127.0.0.1", 9))
        s = socket.socket()
        try:
            with pytest.raises(NetworkBlockedError):
                s.connect(("127.0.0.1", 9))
        finally:
            s.close()


# ---- leakage ---------------------------------------------------------------------------------

POLICY_SIDE = [
    PKG / "planner.py",
    PKG / "controller.py",
    PKG / "prompting.py",
    PKG / "verifier.py",
    PKG / "profiler.py",
    ROOT / "training" / "sampler.py",
    ROOT / "training" / "constraint.py",
    ROOT / "training" / "ablation" / "actions.py",
    ROOT / "training" / "ablation" / "features.py",
    ROOT / "training" / "ablation" / "policy.py",
    PKG / "protocol.py",
    PKG / "llm.py",
    PKG / "policy_review.py",
]


@pytest.mark.parametrize("path", POLICY_SIDE, ids=lambda p: p.name)
def test_policy_side_never_mentions_keys(path: Path):
    calls = _calls(path)
    bad = {c for c in calls if c.startswith("import:") and ("keys" in c or c.startswith("import:benchmark"))}
    assert not bad, f"{path.name} imports answer-key modules: {bad}"
    text = path.read_text(encoding="utf-8")
    assert "keys.jsonl" not in text and "load_keys" not in text


def test_import_graph_has_no_keys():
    code = (
        "import sys; import nanostat.planner, nanostat.controller, nanostat.cli, nanostat.api, "
        "training.sampler, training.constraint, training.ablation.features, "
        "training.ablation.policy, training.ablation.actions; "
        "bad=[m for m in sys.modules if m=='benchmark.keys' or m.startswith('benchmark')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_prompt_contains_no_key_fields(smoke_data: Path):
    import pandas as pd

    from nanostat.profiler import profile
    from nanostat.prompting import render_prompt
    from nanostat.schemas import TaskRecord

    for line in (smoke_data / "tasks" / "train.jsonl").read_text(encoding="utf-8").splitlines()[:20]:
        t = TaskRecord.model_validate_json(line)
        prompt = render_prompt(t.question, t.study, profile(pd.read_csv(smoke_data / t.data_ref)))
        visible = prompt.split("Question:")[1].lower()
        for forbidden in (
            "h0_true",
            "admissible",
            "reference_decision",
            "discordance",
            "perturbation",
            "twin",
            "policy_cell",
            "borderline",
            "skewed",
            "outlier",
        ):
            assert forbidden not in visible, forbidden
        assert "keys" not in t.data_ref
