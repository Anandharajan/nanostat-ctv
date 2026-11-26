# NOTICE

NanoStat-CTV — Copyright 2026 Anandharajan TRV. Licensed under the Apache License 2.0 (see LICENSE).

Third-party material referenced (not redistributed in this repository):

- **P-Bench** (`May2222/P-Bench`, Hugging Face), CC BY 4.0. Not downloaded or used in v0.1; the P-Bench adapter is milestone M6 and has NOT RUN.
- **Qwen2.5-Coder-1.5B-Instruct** (Qwen team), Apache-2.0. Named in `configs/t4_ctv.yaml`; not downloaded or used in v0.1.
- **Fisher-R1** (Miao, Mu, Chen, Zou). The comparator reward (`training/reward_server.py::score_fisher`) is a clean-room implementation from the paper's description; no Fisher-R1 code or data is included.

Runtime dependencies (installed by pip, not vendored): numpy, pandas, scipy, statsmodels, pydantic, PyYAML, typer, FastAPI, uvicorn, Jinja2 — each under its own licence.
