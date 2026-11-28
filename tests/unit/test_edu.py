"""Teaching track (edu/nanostat.py): statistics vs scipy, reward ordering, and a tiny training smoke test."""

import importlib.util
import pathlib

import numpy as np
import pytest
from scipy import stats as S

pytest.importorskip("torch")

_PATH = pathlib.Path(__file__).resolve().parents[2] / "edu" / "nanostat.py"
_spec = importlib.util.spec_from_file_location("edu_nanostat", _PATH)
ns = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ns)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tests_match_scipy(seed):
    rng = np.random.default_rng(seed)
    a, b = rng.normal(0.4, 1, 23), rng.normal(0, 1.7, 31)
    d = rng.normal(0.3, 1, 27)
    assert ns.student_t(a, b, "two-sided") == pytest.approx(S.ttest_ind(a, b).pvalue, abs=1e-10)
    assert ns.welch_t(a, b, "two-sided") == pytest.approx(S.ttest_ind(a, b, equal_var=False).pvalue, abs=1e-10)
    assert ns.welch_t(a, b, "greater") == pytest.approx(
        S.ttest_ind(a, b, equal_var=False, alternative="greater").pvalue, abs=1e-10
    )
    assert ns.paired_t(d, "less") == pytest.approx(S.ttest_1samp(d, 0, alternative="less").pvalue, abs=1e-10)
    x, y = a, a * 0.3 + rng.normal(0, 1, 23)
    assert ns.pearson(x, y, "two-sided") == pytest.approx(S.pearsonr(x, y)[1], abs=1e-10)
    assert ns.spearman(x, y, "two-sided") == pytest.approx(S.spearmanr(x, y)[1], abs=1e-10)
    assert ns.mann_whitney(a, b, "two-sided") == pytest.approx(
        S.mannwhitneyu(a, b, method="asymptotic", use_continuity=True).pvalue, abs=1e-10
    )
    assert ns.wilcoxon(d, "two-sided") == pytest.approx(
        S.wilcoxon(d, correction=True, method="approx").pvalue, abs=1e-10
    )


def test_ranks_average_ties():
    assert ns.ranks(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]


def test_reward_ordering():
    """admissible > not-in-A >= 0 > invalid > malformed (CTV); Fisher-style pays invalid methods."""
    t = ns.make_task(np.random.default_rng(0), world=("two_group", "outliers"), effect="small")
    good, bad, broken = t["gold"], ["two_group", "student_t", t["alt"]], ["paired", "wilcoxon", t["alt"]]
    assert ns.ctv_reward(t, good) >= 0.5
    assert ns.ctv_reward(t, bad) == -0.2
    assert ns.ctv_reward(t, broken) == ns.MALFORMED < -0.2
    assert ns.fisher_reward(t, good) == pytest.approx(1.0)
    assert ns.fisher_reward(t, bad) > 0.0
    assert ns.execute(t, broken) is None


def test_answer_keys_are_consistent():
    for (kind, _), (A, B) in ns.RULES.items():
        assert A and not set(A) & set(B)
        assert all(ns.METHODS[m][0] == kind for m in A)


def test_model_size_and_grammar():
    agent = ns.Agent(ns.build_vocab())
    rng = np.random.default_rng(1)
    plans = agent.decode(agent.act([agent.encode(ns.make_task(rng)["obs"]) for _ in range(16)]))
    for p in plans:
        assert p[0] in ns.SLOTS[0] and p[1] in ns.SLOTS[1] and p[2] in ns.SLOTS[2]


def test_sft_and_grpo_smoke():
    rng = np.random.default_rng(0)
    vocab = ns.build_vocab()
    train = [ns.make_task(rng) for _ in range(200)]
    agent = ns.Agent(vocab)
    ns.sft(agent, train[:40], steps=20)
    used = ns.grpo(agent, train, "ctv", rollouts=640)
    assert used == 640  # exact rollout budget: 10 steps x 8 tasks x 8 samples
    ev = ns.evaluate(agent, train[:50])
    assert 0.0 <= ev["validity"] <= 1.0
