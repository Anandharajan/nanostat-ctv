# STATUS
Milestone: M1 (CPU) complete except second-machine bundle check · Release: v0.1.0 · Date: 2026-11-26

## Done this session
- M0: repo skeleton (SPEC §12), §5 schemas (Pydantic v2, `extra="forbid"`), preprocess DSL, 7 M0 tools + 5 M1 tools + Yuen trimmed t, profiler, diagnostics, verifier, rule planner (B0), controller with LLM-planner interface (mock-tested), CLI (`analyze`, `bundle`, `factory`, `eval`, `train --stage sft-data`, `serve`), local FastAPI.
- M1: policy registry (17 cells, all `reviewed_by: PENDING`), task factory, reference solver, discordance D(x), 6 twin edits (wording-fixed + wording-changed; `outcome_type` and `small_counts` held out for test), lineage splits, key store, minimal audit bundle (hash-chained events, standalone `analysis.py`, manifest).
- Ahead of milestone (CPU-only, unit-tested, never used in training): CTV reward (§7.4), Fisher-style comparator reward, validity constraint (§7.6), twin-joint advantage (§7.5), discordance sampler, programmatic SFT data builder, NanoStat-Bench metrics, agent-level null calibration.
- Bugs found by the pilot runs and fixed with regression tests: χ² crash on an all-zero outcome column; constant binary outcome fell through to Welch with p = NaN; outlier DGPs contaminated one group / one direction so "null" scenarios were not true nulls.

## Commands run (exact)
- `pytest -p no:cacheprovider` → 151 passed
- `ruff check .` → All checks passed; `ruff format --check .` → clean; `mypy` → no issues in 36 source files
- Clean venv: `pip install .` (non-editable wheel), then `nanostat version`, `nanostat analyze ... --rules-only`, `nanostat factory --profile smoke` from outside the repo → OK (policies load from packaged `_policies`, 17 cells)
- `nanostat factory --profile pilot --out <tmp>/pilot` → tasks=1400 twin_pairs=200 splits={train: 1026, test: 242, dev: 132} rejected=49 duplicates=0 (29.5 s)
- `nanostat eval --arms B0 --suite nanostat --data <tmp>/pilot --split dev|test`
- `nanostat eval --arms B0 --suite nullcal --sims 1000` (3 min 19 s)

## Measured (never estimated)
- B0 dev: method validity 127/127, invalid 0/127, twin consistency 15/15, abstention 5/5, operational failures 0/132.
- B0 test: method validity 239/239, invalid 0/239, twin consistency 74/74, abstention 3/3, operational failures 0/242.
  - Circular by construction (keys and B0 share registry + diagnostics); consistency check only.
- D(x) histogram (pilot, 10 bins from 0.0): 879, 128, 74, 29, 17, 67, 32, 44, 41, 89.
- B0 agent-level FPR (1000 sims each): see README table. Liberal: two-group clean small 0.065 [0.051, 0.082]; correlation with leverage points 0.072 [0.057, 0.090]. Others within or below 0.05 ± CI.

## NOT RUN
- M1 exit item "ZIP reproduces on a second machine": tested only in a fresh isolated Python process on the same machine.
- Python–R parity: `Rscript` not installed → `parity.json` reports NOT RUN.
- M2–M8: base model, SFT, CTV-GRPO, arms B1/S0/R-*, P-Bench, M4 plugins, clustering twins, pilot users.
- CI on GitHub Actions: workflow committed; first run result not yet observed at time of writing.

## Blockers
- GPU (T4) needed for M2+. Statistician review needed before any confirmatory run (all policy cells PENDING).

## Next action
- Observe the first GitHub Actions run; fix anything platform-specific (Linux, Python 3.10).
- Second-machine bundle reproduction (M1 exit).
- M2: HF loader (NF4), typed-plan prompting, programmatic SFT on a T4; measure VRAM and tokens/s.

## Artifact hashes (pilot, seed 20261126, generator 0.1.0)
- tasks/train.jsonl: sha256:51aee801870745328b586affa87b6607abbda5b2e18c90e9f84077093a18a234
- tasks/dev.jsonl: sha256:fb049b8a0c02299a4351675ae952c732f4183d6f3cc41bb393cdf7e8d3f3eb0c
- tasks/test.jsonl: sha256:b9afbdea975b99c0992fcd211244269c31587915b6ef6f04a6238b2a73eafdba
- keys/keys.jsonl: sha256:063df7e9217bf62d4a7a97594879aa31336767f6ea252f577318391fddf81dff
- manifests/split_manifest.csv: sha256:e2748558a6e0dafdeda1e10af399fd07959b10495f3ed90c47497b760524906e
