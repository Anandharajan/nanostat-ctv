# Validating real LLMs locally with Magnitude

[Magnitude](https://github.com/magnitudedev/magnitude) (Apache-2.0) is a local inference engine. It runs GGUF models from Hugging Face on your own machine, spreading them over GPU VRAM, RAM and SSD, and serves them through an **OpenAI-compatible API at `http://127.0.0.1:10100/inference/v1`** (no API key; docs: https://docs.magnitude.dev). NanoStat talks to it with a standard-library client (`nanostat/llm.py`) that **refuses non-local endpoints**, so study data never leave the machine.

Magnitude is **inference only**. It runs the validation arms below; it does not train models (SFT/GRPO need a CUDA GPU; see `notebooks/02_train_t4.ipynb`).

## 1. Install and serve (you run these)
Install from the official releases page or installer (desktop app, or the CLI script on the Magnitude site), then:

```powershell
magnitude models search qwen2.5-coder
magnitude models pull <model-id>        # e.g. a Qwen2.5-Coder-1.5B-Instruct GGUF
magnitude serve
magnitude models list                   # note the exact model id
```

Suggested models (check `magnitude models families` / `recommend` for what your hardware supports):

| Model | Why |
|---|---|
| Qwen2.5-Coder-1.5B-Instruct | NanoStat's planned backbone → arm **B1** (base model + typed tools) |
| Qwen2.5-Coder-7B-Instruct | Fisher-R1's base backbone → a same-harness reference for the 7B base |
| One recent small general model (e.g. a Qwen3 or Llama 3.x 3–8B) | sanity check that results are not model-specific |

## 2. Validation runs (CPU-side harness, model on Magnitude)
```powershell
nanostat eval --arms B1 --llm-model <id> --data data --split dev --max-steps 3
nanostat eval --arms B1 --llm-model <id> --data data --split test --max-steps 1   # single shot
nanostat eval --arms B1 --suite nullcal --llm-model <id> --sims 100               # agent-level FPR with the LLM planner
nanostat eval --arms B1 --suite pbench --llm-model <id> --trials 3 --limit 50     # pilot exposure (disclose)
nanostat eval --arms B1 --suite pbench --llm-model <id> --trials 3                # full 425 x 3, frozen configuration only
nanostat review-policies --llm-model <id>                                          # policy PRE-review worksheet
nanostat analyze -q "..." --csv data.csv --study study.json --llm-model <id>      # use an LLM planner yourself
```

Every run writes a resumable JSONL trace (`results/…`) with the raw model outputs, prompt version and hash, attempts, verifier errors and the executed result. Metrics: first-attempt JSON validity, first-attempt invalid-method rate, method validity, Strict (single reference), twin consistency, abstention correctness, operational failures, seconds per task; for P-Bench, the clean-room Raw/Strict/Relaxed pass@1/pass@3 by difficulty and category.

## 3. Rules
- `--data data` must hold a factory output (`nanostat factory --profile pilot --out data`).
- No tuning on P-Bench or the locked test split; tune prompts only on dev. P-Bench pilot exposure (≤ 50 tasks) must be disclosed.
- The **policy pre-review is not a statistician sign-off**; `reviewed_by` stays PENDING until a named statistician signs.
- Report every run, including failures; numbers are "measured on <model id>, Magnitude <version>, prompt <hash>".

## Status
Magnitude is **not installed** on the development machine at the time of writing, so **no LLM result exists yet (NOT RUN)**. The client, harness, P-Bench runner and pre-review are tested against a mock local OpenAI-compatible server (`tests/unit/test_llm_pbench_m4.py`).
