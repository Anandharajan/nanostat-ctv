# Leaderboard facts

**Status: as of 2026-11-27, from the maintainer's master plan; NOT re-verified in this session** (no web checks were made). Re-check each row before acting on it.

| Platform | Facts (as reported) | Implication for NanoStat-CTV |
|---|---|---|
| BenchmarkList, P-Bench page | 11 models; ranked by a composite equal to Hard Strict pass@1; every entry "Self-reported", sourced from arXiv; no submission procedure, only a Feedback link; last updated 28 Aug 2026 | Realistic path: arXiv preprint with a full 425 × 3 table in the Fisher-R1 format, then the Feedback link with an evidence pack. The listing would read "Self-reported". |
| P-Bench on Hugging Face | CC BY 4.0; no leaderboard; no `eval.yaml` | Ask the authors to register it as an HF benchmark; until then, results live only in the model card. |
| HF Community Evals | Benchmarks self-register via `eval.yaml` (beta allow-list); results self-reported unless verified | NanoStat-Bench (twins, null calibration) can be registered as our own benchmark; that is not evidence of rank. |
| Fisher-R1 repository | No licence file; has evaluation code; 7B and 14B weights released; training data not released | Read only. Clean-room scorer (done: `benchmark/pbench/score.py`). Fisher-R1-7B can be re-run on our harness. |
| Kaggle / Codabench | Kaggle: compute and public notebooks, not a ranking for a local LoRA agent. Codabench needs an independent organiser. | Use Kaggle GPUs; Codabench only with a non-author custodian. |

**Every public step (arXiv, public HF, leaderboard entries) is disclosure. Confirm the IP gate first.**
