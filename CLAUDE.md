# CLAUDE.md — NanoStat-CTV

You are implementing SPEC.md. It is the acceptance contract.

Rules
1. Work on the current milestone in STATUS.md only. Do not skip ahead.
2. CPU first. Run `pytest -q` before and after every change; paste the real summary into STATUS.md.
3. The model never computes statistics. All numbers come from nanostat/tools.
4. Never let planner/controller/training code import from `keys/` or `benchmark/keys.py`. The leakage test must stay green.
5. No eval/exec/shell on model output. R runs only through fixed templates in nanostat/parity/r_backend.py.
6. Never invent results, scores, citations, platform statuses or verification badges. Unrun = NOT RUN.
7. No new dependency, network call, upload or publish action without asking the maintainer.
8. Commits and PRs: author is Anandharajan only. Do not add Claude as co-author or contributor; no Co-Authored-By or "Generated with" lines.
9. Small commits, one concern each, message `M<k>: <what>`.
10. When a test fails, fix the cause; do not loosen tolerances or delete tests without recording why in STATUS.md.
11. Keep notebooks thin: they call package code only.
12. End every session by updating STATUS.md (done, commands, measured numbers, blockers, next action).

First task: M0 — create the repo skeleton from SPEC.md §12, implement §5 schemas, the seven M0 tools with fixtures, the verifier tests in §13, the rule planner and `nanostat analyze --rules-only`. Stop when CI is green and report.

Permanent policy (v3.1)
- Commits/PRs are authored by Anandharajan only; never add Claude as co-author or contributor.
- Fisher-R1 code has no licence: read only, implement clean-room.
- Any download, upload, email, submission or publication requires explicit maintainer approval.
- Before any public release, confirm the IP gate (provisional patent filed or decision not to file).
