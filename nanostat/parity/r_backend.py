"""Python vs R parity via fixed Rscript templates (SPEC §6.1, §10).

This is the ONLY module allowed to start a subprocess. Model output never reaches it:
the R code is chosen from fixed templates by method name, and data are passed as a CSV
file path argument, never interpolated into code.
"""

from __future__ import annotations

import shutil
import subprocess  # noqa: S404 - sole sanctioned use, see module docstring
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nanostat.controller import RunRecord

# Templates read args: csv path, then role columns; print the p-value only.
R_TEMPLATES: dict[str, str] = {
    "welch_t": "a<-commandArgs(TRUE);d<-read.csv(a[1]);g<-factor(d[[a[3]]],levels=c(a[4],a[5]));"
    "cat(t.test(d[[a[2]]][g==a[5]],d[[a[2]]][g==a[4]],var.equal=FALSE,alternative=a[6])$p.value)",
    "student_t": "a<-commandArgs(TRUE);d<-read.csv(a[1]);g<-factor(d[[a[3]]],levels=c(a[4],a[5]));"
    "cat(t.test(d[[a[2]]][g==a[5]],d[[a[2]]][g==a[4]],var.equal=TRUE,alternative=a[6])$p.value)",
    "pearson": "a<-commandArgs(TRUE);d<-read.csv(a[1]);"
    "cat(cor.test(d[[a[2]]],d[[a[3]]],method='pearson',alternative=a[4])$p.value)",
}

_R_ALT = {"two-sided": "two.sided", "greater": "greater", "less": "less"}


def rscript_available() -> bool:
    return shutil.which("Rscript") is not None


def _args(rec: RunRecord) -> list[str] | None:
    plan = rec.plan
    if plan is None or plan.method not in R_TEMPLATES:
        return None
    alt_val = rec.result.alternative.value if rec.result.alternative else "two-sided"
    alt = _R_ALT[alt_val]
    c = plan.columns
    if plan.method in ("welch_t", "student_t"):
        order = rec.study.group_order
        if not order:
            return None
        return [c["y"], c["group"], order[0], order[1], alt]
    if plan.method == "pearson":
        return [c["x"], c["y"], alt]
    return None


def run_r(method: str, csv_path: Path, args: list[str], timeout: int = 60) -> float:
    script = R_TEMPLATES[method]
    exe = shutil.which("Rscript")
    if exe is None:
        raise RuntimeError("Rscript not found")
    out = subprocess.run(  # noqa: S603 - fixed template, list args, no shell
        [exe, "-e", script, str(csv_path), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return float(out.stdout.strip())


def parity_check(rec: RunRecord, data_csv: bytes | None = None) -> dict[str, Any]:
    method = rec.plan.method if rec.plan else None
    if not rscript_available():
        return {"status": "NOT RUN", "reason": "Rscript not available on this machine", "method": method}
    args = _args(rec)
    if args is None or data_csv is None:
        return {"status": "NOT RUN", "reason": "no R template for this method/layout", "method": method}
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "data.csv"
        p.write_bytes(data_csv)
        try:
            r_p = run_r(method or "", p, args)
        except (subprocess.SubprocessError, ValueError, RuntimeError) as e:
            return {"status": "FAILED", "reason": str(e), "method": method}
    py_p = rec.result.p_value
    ok = py_p is not None and abs(r_p - py_p) <= 1e-6 * max(1.0, abs(py_p))
    return {"status": "PASS" if ok else "MISMATCH", "method": method, "python_p": py_p, "r_p": r_p}
