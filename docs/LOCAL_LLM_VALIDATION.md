# Validating real LLMs locally with Magnitude

[Magnitude](https://github.com/magnitudedev/magnitude) (Apache-2.0) is a local inference engine. It runs GGUF models from Hugging Face on your own machine, spreading them over GPU VRAM, RAM and SSD, and serves them through an **OpenAI-compatible API at `http://127.0.0.1:10100/inference/v1`** (no API key; docs: https://docs.magnitude.dev). NanoStat talks to it with a standard-library client (`nanostat/llm.py`) that **refuses non-local endpoints**, so study data never leave the machine.

Magnitude is **inference only**. It runs the validation arms below; it does not train models (SFT/GRPO need a CUDA GPU; see `notebooks/02_train_t4.ipynb`).

**Colab alternative (no local install):** [`notebooks/03_llm_b1_colab.ipynb`](https://colab.research.google.com/github/Anandharajan/nanostat-ctv/blob/main/notebooks/03_llm_b1_colab.ipynb). It serves the model with vLLM *inside* the Colab VM (still a local endpoint for NanoStat's client) and runs B1 on the dev split, with resumable traces you can download.

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

## Status (2026-11-28)
**First LLM result: B1 on the dev split, measured on Colab** (not Magnitude): 0/92 runnable plans with prompt v2. See docs/LLM_B1_BASELINE.md. Everything else with an LLM is still NOT RUN. The client, harness, P-Bench runner and pre-review are tested against a mock local OpenAI-compatible server (`tests/unit/test_llm_pbench_m4.py`).

What happened on the development machine (i7-8750H, 32 GB RAM, Quadro P1000 4 GB):
- Magnitude CLI **0.3.7** installed with the maintainer's approval (official script, checksum and publisher signature verified; CLI only).
- Magnitude 0.3.7 **does not run the `qwen2` or `qwen3` architectures** ("Magnitude doesn't run qwen2 yet"), so the planned backbone `Qwen2.5-Coder-1.5B-Instruct` and the 7B reference cannot be served by it. It runs Qwen3.5.
- Its CUDA path needs an Ampere GPU or newer; the P1000 is used through Vulkan.
- `unsloth/Qwen3.5-2B-GGUF` Q4_K_M was downloaded (listed as 1.95 GB including the vision projector). Magnitude **refused to load it**: "needs 601.0 MB more memory than this computer can provide, even for a 32K context". Windows reported 16.9 GB of RAM free at that moment, but Magnitude counted 7.9 GB. Its smallest context is 32K, and `models load` has no context option. Every quant of the model, down to 768 MB, was marked "too large" after its hardware assessment.
- **Next:** run `notebooks/03_llm_b1_colab.ipynb` on a free Colab T4. It serves Qwen2.5-Coder-1.5B-Instruct with vLLM on 127.0.0.1 inside the VM and runs B1 on the dev split. Or retry Magnitude when it supports qwen2, or on a machine with more free memory.
