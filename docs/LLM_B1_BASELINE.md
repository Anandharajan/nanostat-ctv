# First LLM baseline (arm B1): Qwen2.5-Coder-1.5B-Instruct, prompt v2

**Measured on 2026-11-28** with `notebooks/03_llm_b1_colab.ipynb` on a free Colab Tesla T4: vLLM 0.34.0, torch 2.14.0, transformers 5.22.1, repo commit `d6b0c03`, prompt `typed-plan-v2`, temperature 0. The model was served on 127.0.0.1 inside the Colab VM. Data: pilot factory (seed 20261126, current generator), **dev split, 92 tasks** (88 answerable, 4 clarify/abstain), `data/tasks/dev.jsonl` sha256 prefix `b0c172ceaf369a51`. Raw traces, summaries, versions and the vLLM log are in `results/llm_b1/` (zip SHA-256 `45ea24949d34dc11e4ca814ceb314c985d081b82b5767063f8e0bdb91acabd6e`). The scores below were recomputed locally from the traces with `evaluation.llm_eval.score_traces`.

## Result

| Setting | Runnable plans | Method validity | Invalid-method rate | Twin consistency | Abstention correct | LLM calls | Seconds per task |
|---|---|---|---|---|---|---|---|
| Single shot (`--max-steps 1`) | **0 / 92** | 0 / 88 | 0 / 88 | 0 / 13 | 0 / 4 | 92 | 3.0 |
| Re-plan with feedback (`--max-steps 3`) | **0 / 92** | 0 / 88 | 0 / 88 | 0 / 13 | 0 / 4 | 276 | 8.1 |

Operational failure rate is 1.0 in both settings. The untrained 1.5B model, with this prompt, cannot yet produce a single plan that the schema accepts. The 0% invalid-method rate is **not** good news: nothing ran. It is exactly the trap from the teaching track (lesson 4, demo panel 4). This is why every report must show the broken-plan or operational-failure rate next to the invalid rate.

## Why every plan failed (first attempt, single shot)

All 92 outputs were wrapped in a ```json fence; the parser handles that. The model chose `"action": "clarify"` on 91 of 92 tasks. Schema errors (one plan can have several):

| Error | Plans | What the model did |
|---|---|---|
| `columns` | 50 | used dataset column names as role keys (`{"fertiliser": "fertiliser"}`) instead of roles (`y`, `group`, `x`, `pair_id`, …) |
| `estimand` | 34 | wrote the protocol's estimand **family** (e.g. `"typical"`) instead of an allowed estimand |
| `reason_code` missing | 32 | chose `clarify` but gave no `reason_code` (all 32 structural errors are exactly this) |
| `checks` | 13 | invented check names (`population_filter` ×5, `repeated_measures` ×3, `time_censoring`, `k_groups`, `exposure`, …) |

Re-planning did not help: attempts 2 and 3 repeated the same errors. That is expected from how the harness works now. After a schema failure, the controller's only feedback is *"output was not a valid typed plan JSON"* (`nanostat/controller.py`), with no field and no allowed values.

## What this does and does not show

- **Shows:** the starting point that SFT and GRPO must improve on, measured honestly. It also shows that prompt v2 is under-specified for a small model. It lists role names, estimands and check names as placeholders (`<role>`, `<optional>`, `<name>`) instead of enumerating them, while the locked protocol in the same prompt names an estimand family that looks like a valid value.
- **Does not show:** the capability of Qwen2.5-Coder-1.5B on the task. Most failures are format and specification failures that a clearer prompt or SFT would likely remove. This is not yet measured.
- **No numbers on the locked test split, null calibration or P-Bench** were produced. These stay NOT RUN.

## Next steps (dev split only, disclosed)

1. **Prompt v3:** enumerate the allowed column roles per design, the estimands for the locked family, the check names and reason codes. Make clarify and abstain conditional on stated missing facts. Ask for raw JSON without a fence. Re-run B1 on dev and report v2 and v3 side by side.
2. **Informative feedback:** pass the schema error fields and allowed values back to the model on re-plan. Measure `--max-steps 3` again.
3. Then SFT (`notebooks/02_train_t4.ipynb`) and GRPO with the v0.4.1 reward, tracking the operational-failure rate alongside validity and the invalid rate.
