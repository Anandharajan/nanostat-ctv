"""Evaluator-only P-Bench preflight (SPEC v3.1 Step 3.2).

Derives the 17 categories and counts from the pinned data, asserts 425 / 203 Easy / 222 Hard, and
writes manifests/pbench_method_mapping.csv plus a coverage report:

  * category ceiling   : tasks whose category has a NanoStat plugin (an upper bound);
  * spec-flag ceiling  : of those, tasks whose question text does not mention specification features
                         the typed plan / DSL cannot yet express (constructed weights, stacking,
                         fixed effects, lags, interactions, ...). Keyword heuristic; reported as such.

The agent never sees categories, answers or paths; this module is evaluator-side only.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmark.pbench.loader import PINNED_REVISION, local_dir
from nanostat.tools import REGISTRY

# category -> (required estimator / specification, NanoStat plugin(s) able to produce it)
CATEGORY_PLUGINS: dict[str, tuple[str, tuple[str, ...]]] = {
    "t_test": ("Welch/Student/paired t", ("welch_t", "student_t", "paired_t")),
    "spearman_correlation": ("Spearman rank correlation", ("spearman",)),
    "ols_coefficient_test": ("OLS coefficient t/z test, classical/HC/cluster SE, covariates", ("ols_coef",)),
    "fisher_exact_test": ("Fisher exact test", ("fisher_exact",)),
    "log_rank_test": ("log-rank test", ("log_rank",)),
    "logistic_regression": ("logistic regression Wald test", ("logistic",)),
    "mann_whitney_test": ("Mann-Whitney U", ("mann_whitney",)),
    "iv_2sls": ("2SLS coefficient test (classical/robust/cluster)", ("iv_2sls",)),
    "cox_regression": ("Cox PH coefficient test", ("cox_ph",)),
    "f_test": ("joint/linear restriction F or Wald test after OLS", ("ols_wald",)),
    "probit": ("probit coefficient test", ("probit",)),
    "other": ("heterogeneous (stacked / custom estimators)", ()),
    "mixed_effects": ("linear mixed model fixed-effect test", ("mixed_lm",)),
    "kruskal_wallis_test": ("Kruskal-Wallis", ("kruskal_wallis",)),
    "chi_square_test": ("Pearson chi-squared", ("chi2_independence",)),
    "binomial_test": ("exact binomial", ("binomial_test",)),
    "tobit": ("Tobit (censored) regression", ("tobit",)),
}

SPEC_FLAGS = {
    "constructed_weights": r"\bweight|poids|reweight",
    "stacked_or_reshaped": r"\bstack|pooled dataset|reshap|long format",
    "fixed_effects": r"fixed effect|\bfe\b|strata dummies|dummies for",
    "lags_or_differences": r"\blag|first[- ]difference|\bdiff\b",
    "interaction": r"interaction|\bx\b.*\binteract",
    "to_be_constructed": r"to be constructed|construct(ed)? as|derive[sd]? as",
    "subset_restriction": r"restrict(ed|ion)? to|analysis sample|subsample|only (students|patients|respondents)",
}


def load_tasks(root: Path | None = None) -> list[dict[str, Any]]:
    d = root or local_dir()
    splits = json.loads((d / "splits.json").read_text(encoding="utf-8"))
    out = []
    for key, diff in sorted(splits.items()):
        cat, name = key.split("/", 1)
        tdir = d / "tasks" / cat / name
        out.append(
            {
                "key": key,
                "category": cat,
                "difficulty": diff.lower(),
                "dir": tdir,
                "question": (tdir / "question.txt").read_text(encoding="utf-8"),
            }
        )
    return out


def spec_flags(question: str) -> list[str]:
    q = question.lower()
    return [k for k, pat in SPEC_FLAGS.items() if re.search(pat, q)]


def preflight(out_dir: Path = Path("manifests"), root: Path | None = None) -> dict[str, Any]:
    tasks = load_tasks(root)
    n = len(tasks)
    diff = Counter(t["difficulty"] for t in tasks)
    assert n == 425, f"expected 425 tasks, found {n}"
    assert diff["easy"] == 203 and diff["hard"] == 222, f"difficulty split {dict(diff)}"
    cats = Counter(t["category"] for t in tasks)
    assert len(cats) == 17, f"expected 17 categories, found {len(cats)}"

    by_cat: dict[str, dict[str, Any]] = defaultdict(lambda: Counter())  # type: ignore[arg-type]
    covered = covered_clean = 0
    covered_by_diff: Counter[str] = Counter()
    clean_by_diff: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    for t in tasks:
        req, plugins = CATEGORY_PLUGINS.get(t["category"], ("unknown", ()))
        have = bool(plugins) and all(p in REGISTRY for p in plugins[:1])
        flags = spec_flags(t["question"])
        for f in flags:
            flag_counts[f] += 1
        c = by_cat[t["category"]]
        c["n"] += 1
        c[t["difficulty"]] += 1
        if have:
            covered += 1
            covered_by_diff[t["difficulty"]] += 1
            c["supported"] += 1
            if not flags:
                covered_clean += 1
                clean_by_diff[t["difficulty"]] += 1
                c["supported_no_flags"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pbench_method_mapping.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "category",
                "n",
                "easy",
                "hard",
                "required_estimator",
                "plugins",
                "status",
                "supported_tasks",
                "supported_without_spec_flags",
            ]
        )
        for cat, cnt in sorted(cats.items(), key=lambda kv: -kv[1]):
            req, plugins = CATEGORY_PLUGINS.get(cat, ("unknown", ()))
            status = (
                "supported" if plugins and plugins[0] in REGISTRY else ("missing" if plugins else "not_expressible")
            )
            c = by_cat[cat]
            w.writerow(
                [
                    cat,
                    cnt,
                    c["easy"],
                    c["hard"],
                    req,
                    "|".join(plugins),
                    status,
                    c["supported"],
                    c["supported_no_flags"],
                ]
            )
    report = {
        "revision": PINNED_REVISION,
        "tasks": n,
        "easy": diff["easy"],
        "hard": diff["hard"],
        "categories": dict(cats),
        "category_ceiling": {
            "all": covered / n,
            "easy": covered_by_diff["easy"] / diff["easy"],
            "hard": covered_by_diff["hard"] / diff["hard"],
            "tasks": covered,
        },
        "spec_flag_ceiling": {
            "all": covered_clean / n,
            "easy": clean_by_diff["easy"] / diff["easy"],
            "hard": clean_by_diff["hard"] / diff["hard"],
            "tasks": covered_clean,
        },
        "spec_flag_counts": dict(flag_counts),
        "note": "spec flags are a keyword heuristic over question text; ceilings are upper bounds, not accuracy",
    }
    (out_dir / "pbench_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
