"""`nanostat` command-line interface (SPEC §11)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import typer

from nanostat import __version__
from nanostat.controller import run_analysis
from nanostat.runs import bundle_path, save_run
from nanostat.schemas import Result, Study
from nanostat.security import InputError, read_csv_limited

app = typer.Typer(add_completion=False, help="NanoStat-CTV: local, auditable hypothesis testing.")

NOT_AVAILABLE = (
    "{what} is not available in v{v}: it needs the GPU stages (milestones M2-M5), which have NOT RUN. See STATUS.md."
)


def _print_result(r: Result, rid: str | None) -> None:
    typer.echo(f"action      : {r.action.value}")
    if r.reason_code:
        typer.echo(f"reason      : {r.reason_code.value}")
    if r.method:
        typer.echo(
            f"method      : {r.method}  [{r.method_status.value}"
            + (f": {r.status_reason}" if r.status_reason else "")
            + "]"
        )
    if r.p_value is not None:
        typer.echo(f"design      : {r.design.value if r.design else '-'}  ({r.policy_cell})")
        typer.echo(f"alternative : {r.alternative.value if r.alternative else '-'}")
        typer.echo(f"statistic   : {r.statistic:.6g}")
        typer.echo(f"p-value     : {r.p_value:.6g}   (alpha = {r.alpha}) -> {r.decision}")
        if r.effect is not None:
            typer.echo(f"effect      : {r.effect:.6g}  ({r.effect_label})")
        if r.ci is not None:
            typer.echo(f"CI          : [{r.ci[0]:.6g}, {r.ci[1]:.6g}]")
        typer.echo(f"n           : {r.n}")
        if r.admissible_range:
            typer.echo("admissible methods on your data (sensitivity range):")
            for row in r.admissible_range:
                p = "error: " + (row.error or "") if row.p is None else f"p = {row.p:.6g}"
                typer.echo(f"  - {row.method:22s} {p}")
    for msg in r.messages:
        typer.echo(f"note        : {msg}")
    if rid:
        typer.echo(f"run id      : {rid}   (nanostat bundle --run {rid} --out bundle.zip)")


@app.command()
def analyze(
    question: str = typer.Option(..., "--question", "-q", help="Research question in plain language."),
    csv: Path = typer.Option(..., "--csv", exists=True, dir_okay=False, help="Data file (CSV)."),
    study: Optional[Path] = typer.Option(None, "--study", exists=True, dir_okay=False, help="Study metadata JSON."),
    adapter: Optional[Path] = typer.Option(None, "--adapter", help="Trained LoRA adapter (not available in v0.1)."),
    rules_only: bool = typer.Option(False, "--rules-only", help="Use the deterministic rule planner (B0)."),
    json_out: bool = typer.Option(False, "--json", help="Print the full result JSON."),
    diagnostic: bool = typer.Option(
        False, "--diagnostic", help="Allow methods outside the admissible set; returns p-values but no decision."
    ),
) -> None:
    """Plan, verify and run a hypothesis test on a CSV."""
    if adapter is not None and not rules_only:
        typer.echo(NOT_AVAILABLE.format(what="--adapter (LLM planner)", v=__version__), err=True)
        raise typer.Exit(2)
    data = csv.read_bytes()
    try:
        df = read_csv_limited(data)
    except InputError as e:
        typer.echo(f"input error: {e}", err=True)
        raise typer.Exit(2) from e
    st = Study.model_validate_json(study.read_text(encoding="utf-8")) if study else Study()
    rec = run_analysis(question, st, df, mode="diagnostic" if diagnostic else "confirmatory")
    rid, _ = save_run(rec, data)
    if json_out:
        typer.echo(
            json.dumps(
                {
                    "run_id": rid,
                    "plan": rec.plan.model_dump(mode="json") if rec.plan else None,
                    "result": rec.result.model_dump(mode="json"),
                },
                indent=2,
                default=str,
            )
        )
    else:
        _print_result(rec.result, rid)


@app.command()
def bundle(run: str = typer.Option(..., "--run"), out: Path = typer.Option(Path("bundle.zip"), "--out")) -> None:
    """Export the audit bundle (ZIP) of a previous run."""
    try:
        src = bundle_path(run)
    except (InputError, FileNotFoundError) as e:
        typer.echo(f"run not found: {run}", err=True)
        raise typer.Exit(2) from e
    shutil.copyfile(src, out)
    typer.echo(f"wrote {out}")


@app.command()
def factory(
    profile: str = typer.Option("smoke", "--profile", help="smoke | pilot | full"),
    out: Path = typer.Option(Path("data"), "--out"),
    seed: int = typer.Option(20261126, "--seed"),
) -> None:
    """Generate synthetic tasks, answer keys, twins and manifests (CPU)."""
    from benchmark.factory.grid import PROFILES, run_factory

    if profile not in PROFILES:
        raise typer.BadParameter(f"profile must be one of {sorted(PROFILES)}")
    rep = run_factory(profile, out, seed)
    typer.echo(
        f"tasks={rep.tasks} twin_pairs={rep.twin_pairs} splits={dict(rep.per_split)} "
        f"rejected={sum(rep.rejected.values())} duplicates={rep.duplicates}"
    )
    typer.echo(f"report: {out / 'manifests' / 'factory_report.json'}")


@app.command()
def eval(  # noqa: A001 - CLI verb from SPEC §11
    arms: str = typer.Option("B0", "--arms"),
    suite: str = typer.Option("nanostat", "--suite", help="nanostat | twins | nullcal | pbench"),
    data: Path = typer.Option(Path("data"), "--data"),
    split: str = typer.Option("dev", "--split"),
    sims: int = typer.Option(1000, "--sims", help="null-calibration simulations per scenario"),
    out: Optional[Path] = typer.Option(None, "--out"),
) -> None:
    """Evaluate arms. v0.1 can evaluate only B0 (rule planner) on CPU."""
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    for a in arm_list:
        if a != "B0":
            typer.echo(NOT_AVAILABLE.format(what=f"arm {a}", v=__version__), err=True)
            raise typer.Exit(2)
    if suite == "pbench":
        typer.echo(NOT_AVAILABLE.format(what="the P-Bench suite (M6/M7)", v=__version__), err=True)
        raise typer.Exit(2)
    if suite == "nullcal":
        from benchmark.nullcal import rows_to_json, run_nullcal

        rows = run_nullcal(sims=sims)
        payload: Any = rows_to_json(rows)
    else:
        from evaluation.metrics import evaluate_split

        summ, _ = evaluate_split(data, split)
        payload = {"arm": "B0", "split": split, "n": summ.n, **summ.metrics}
        if suite == "twins":
            payload = {
                "arm": "B0",
                "split": split,
                "twin_consistency_accuracy": summ.metrics["twin_consistency_accuracy"],
            }
    text = json.dumps(payload, indent=2, default=str)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    typer.echo(text)


@app.command()
def train(
    stage: str = typer.Option(..., "--stage"),
    arm: str = typer.Option("R-CTV", "--arm"),
    config: Path = typer.Option(Path("configs/t4_ctv.yaml"), "--config"),
    data: Path = typer.Option(Path("data"), "--data"),
    out: Path = typer.Option(Path("data/sft.jsonl"), "--out"),
) -> None:
    """Training entry point. v0.1 only builds programmatic SFT data on CPU (--stage sft-data)."""
    if stage == "sft-data":
        from training.sft_data import build_sft

        n = build_sft(data, out)
        typer.echo(f"wrote {n} SFT examples to {out}")
        return
    typer.echo(NOT_AVAILABLE.format(what=f"--stage {stage} ({arm})", v=__version__), err=True)
    raise typer.Exit(2)


@app.command()
def ablate(
    data: Path = typer.Option(Path("data"), "--data", help="Factory output directory (pilot or full)."),
    seeds: str = typer.Option("1,2,3,4,5", "--seeds"),
    updates: int = typer.Option(300, "--updates"),
    null_sims: int = typer.Option(300, "--null-sims", help="fresh null simulations per scenario (0 = skip)"),
    out: Path = typer.Option(Path("results/ablation_cpu"), "--out"),
) -> None:
    """CPU ablation of the CTV training signal with a proxy policy (S0, R-F, R-A, R-AD, R-ADT, R-CTV)."""
    from training.ablation.experiment import HParams, run_ablation, write_report
    from training.ablation.tables import NullTable, TaskTable, build_null_table, build_task_table

    tpath, npath = data / "ablation_tasks.npz", data / "ablation_null.npz"
    tb = TaskTable.load(tpath) if tpath.exists() else build_task_table(data)
    if not tpath.exists():
        tb.save(tpath)
    nt = None
    if null_sims > 0:
        nt = NullTable.load(npath) if npath.exists() else build_null_table(null_sims)
        if not npath.exists():
            nt.save(npath)
    res = run_ablation(tb, nt, [int(s) for s in seeds.split(",")], HParams(updates=updates))
    write_report(res, out)
    typer.echo(f"wrote {out / 'ablation_results.md'}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the local API (localhost only by default)."""
    import uvicorn

    uvicorn.run("nanostat.api:app", host=host, port=port)


@app.command()
def version() -> None:
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
