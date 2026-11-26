"""Local API (SPEC §11). Uploads only (CSV text in the request body); no server paths."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from nanostat import __version__
from nanostat.controller import run_analysis
from nanostat.runs import bundle_path, load_run, save_run
from nanostat.schemas import Study
from nanostat.security import MAX_BYTES, InputError, read_csv_limited

app = FastAPI(title="NanoStat-CTV", version=__version__)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=3, max_length=2000)
    csv: str = Field(max_length=MAX_BYTES)
    study: Study = Field(default_factory=Study)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__, "planner": "rules"}


@app.post("/v1/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    data = req.csv.encode()
    try:
        df = read_csv_limited(data)
    except InputError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    rec = run_analysis(req.question, req.study, df)
    rid, _ = save_run(rec, data)
    return {
        "run_id": rid,
        "plan": rec.plan.model_dump(mode="json") if rec.plan else None,
        "result": rec.result.model_dump(mode="json"),
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return load_run(run_id)
    except (InputError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail="run not found") from e


@app.get("/v1/runs/{run_id}/bundle")
def get_bundle(run_id: str) -> FileResponse:
    try:
        p = bundle_path(run_id)
    except (InputError, FileNotFoundError) as e:
        raise HTTPException(status_code=404, detail="run not found") from e
    return FileResponse(p, media_type="application/zip", filename=f"nanostat_{run_id}.zip")
