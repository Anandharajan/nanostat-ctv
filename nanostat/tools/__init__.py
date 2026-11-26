"""Whitelisted tool registry. Only plugins listed here can ever be executed."""

from __future__ import annotations

from nanostat.tools.anova_oneway import AnovaOneway
from nanostat.tools.base import RunContext, Tool, ToolError, ToolOutput
from nanostat.tools.binomial_test import BinomialTest
from nanostat.tools.chi2_independence import Chi2Independence
from nanostat.tools.fisher_exact import FisherExact
from nanostat.tools.kruskal_wallis import KruskalWallis
from nanostat.tools.mann_whitney import MannWhitney
from nanostat.tools.paired_t import PairedT
from nanostat.tools.pearson import Pearson
from nanostat.tools.spearman import Spearman
from nanostat.tools.student_t import StudentT
from nanostat.tools.welch_t import WelchT
from nanostat.tools.wilcoxon import WilcoxonSignedRank
from nanostat.tools.yuen_trimmed_t import YuenTrimmedT

M0_TOOLS = (
    "welch_t",
    "student_t",
    "paired_t",
    "mann_whitney",
    "wilcoxon_signed_rank",
    "pearson",
    "spearman",
)
M1_TOOLS = ("anova_oneway", "kruskal_wallis", "chi2_independence", "fisher_exact", "binomial_test")
EXTRA_TOOLS = ("yuen_trimmed_t",)

REGISTRY: dict[str, Tool] = {
    t.name: t
    for t in (
        WelchT(),
        StudentT(),
        YuenTrimmedT(),
        PairedT(),
        MannWhitney(),
        WilcoxonSignedRank(),
        Pearson(),
        Spearman(),
        AnovaOneway(),
        KruskalWallis(),
        Chi2Independence(),
        FisherExact(),
        BinomialTest(),
    )
}


def get_tool(name: str) -> Tool:
    try:
        return REGISTRY[name]
    except KeyError as e:
        raise ToolError(f"unsupported method {name!r}") from e


__all__ = [
    "EXTRA_TOOLS",
    "M0_TOOLS",
    "M1_TOOLS",
    "REGISTRY",
    "RunContext",
    "Tool",
    "ToolError",
    "ToolOutput",
    "get_tool",
]
