# First LLM baseline (arm B1): Qwen2.5-Coder-1.5B-Instruct, prompts v2 and v3

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

## Prompt v3 (dev-tuned after the v2 result above; disclosed)

**Measured on 2026-11-28** with the same model, the same dev split (sha256 prefix `b0c172ceaf369a51`, identical tasks), Colab T4, vLLM 0.34.0, repo commit `d5e94d4`, prompt `typed-plan-v3` (hash `fa4acf5922308339`). Traces were written to Google Drive while running; the zip was verified after download (44,762 bytes, SHA-256 `fb761960201a3418fdb27429aadbd43ac1e7a619ed960e4e05530a4cdf30d28d`). Files are `results/llm_b1/qwen25coder15b_v3_dev_steps{1,3}.json[l]`, `versions_v3.txt` and `vllm_v3.log`.

What changed in v3 (`nanostat/prompting.py`): every allowed role, check and reason code is enumerated; the prompt says to copy `columns` from `Study.roles`, to omit `estimand`, and to output raw JSON; execute is the default and clarify needs a named missing fact. Re-plan feedback now names the failing field and the allowed values (`nanostat/planner.py::plan_errors`).

| | v2 single shot | v2 3 steps | **v3 single shot** | **v3 3 steps** |
|---|---|---|---|---|
| Schema-valid first plans | 0 / 92 | 0 / 92 | **16 / 92** | 16 / 92 |
| First-attempt action = execute | 1 / 92 | 1 / 92 | **62 / 92** | 63 / 92 |
| No operational failure (final: executed / valid clarify) | 0 / 92 | 0 / 92 | 12 / 92 (1 / 11) | **20 / 92** (1 / 19) |
| Method validity (answerable) | 0 / 88 | 0 / 88 | 1 / 88 | 1 / 88 |
| Invalid-method rate | 0 / 88 | 0 / 88 | 0 / 88 | 0 / 88 |
| Strict (single reference) | 0 / 88 | 0 / 88 | 1 / 88 | 1 / 88 |
| LLM calls | 92 | 276 | 92 | 244 |
| Seconds per task | 3.0 | 8.1 | 3.5 | 8.3 |

**Reading.** v3 fixed the *intent*: the model now tries to answer, with execute on 62 of 92 first plans instead of 1. It also proposes varied methods (`ols_coef` 17, `paired_t` 12, `wilcoxon_signed_rank` 12, `anova_oneway` 8, `binomial_test`, `chi2_independence`). Field-level feedback helps a little: 3 steps turn 8 more tasks from broken output into a well-formed answer (12 → 20). But those are almost all *clarify* requests; only one task ends in an executed test. Validity is still essentially zero, because:

- **Column roles are still wrong in 51 of 92 first plans.** The model writes column names as keys (`{"fertiliser": "fertiliser", ...}`) or reverses the mapping (`{"section": "group", "score": "y"}`), even with the instruction and an example. This one error blocks most otherwise sensible plans.
- **13 execute plans carry `"reason_code": ""`**, and 10 clarify plans have no reason code at all.
- **Every output is still wrapped in a ```json fence**, despite the instruction. That is harmless: the parser takes the first JSON object.
- Of the 20 tasks without an operational failure (3 steps), 19 end in a clarify request on an answerable task (counted as false abstention) and 1 executes a valid method. Method validity therefore stays at 1 of 88.

**Conclusion.** Prompting alone gets an untrained 1.5B model to the point of *trying*, but not to producing correct, schema-valid plans. The remaining errors are systematic (role mapping, empty fields), which is exactly what supervised fine-tuning on the factory's gold plans teaches. The next step is SFT (`notebooks/02_train_t4.ipynb`), not further prompt tuning against the dev split. A deterministic post-processor that, for example, copies `Study.roles` into `columns` would also lift the numbers. But it would hide the model's real ability, so if it is ever added it must be reported as a separate arm.
