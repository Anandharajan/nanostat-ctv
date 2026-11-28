"""GPU training entry points (Colab / Kaggle T4). NOT RUN in this repository's environment (no CUDA).

    python -m training.train_gpu sft  --data data --sft data/sft.jsonl --out runs/s0
    python -m training.train_gpu grpo --data data --sft data/sft.jsonl --init runs/s0 --arm R-A --out runs/ra

Requires the optional GPU extras: pip install -e ".[gpu]" (torch, transformers, peft, trl, datasets,
accelerate, bitsandbytes). Written against the TRL SFTTrainer / GRPOTrainer APIs; verify against the
pinned TRL version in the smoke notebook before any real run, and record the version in STATUS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sft_records(path: str, limit: int | None = None) -> list[dict]:
    """SFT rows in TRL's conversational prompt-completion format. Evaluation sends the prompt as a single user
    chat message (nanostat.llm.OpenAICompatClient), and the server applies the model's chat template, so
    training must use the same template. Otherwise the model is evaluated on a format it never saw."""
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[:limit] if limit else rows
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "completion": [{"role": "assistant", "content": r["completion"]}],
        }
        for r in rows
    ]


def _load_fp32(model_id: str):  # type: ignore[no-untyped-def]
    """T4 (sm_75) has no bf16: load fp32 master weights and train with fp16 autocast + LoRA."""
    import torch
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    except TypeError:  # older transformers
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)


def sft(args: argparse.Namespace) -> None:
    import time

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    cfg = _cfg(args.config)
    s_cfg = cfg["sft"]
    ds = Dataset.from_list(sft_records(args.sft, s_cfg["examples"]))
    lora = cfg["lora"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trainer = SFTTrainer(
        model=_load_fp32(cfg["model_id"]),
        processing_class=AutoTokenizer.from_pretrained(cfg["model_id"]),
        train_dataset=ds,
        args=SFTConfig(
            output_dir=str(out),
            per_device_train_batch_size=s_cfg["micro_batch"],
            gradient_accumulation_steps=s_cfg["grad_accum"],
            learning_rate=s_cfg["lr"],
            num_train_epochs=args.epochs or s_cfg["epochs"],
            max_length=s_cfg["max_seq"],
            completion_only_loss=True,
            gradient_checkpointing=True,
            fp16=True,
            logging_steps=5,
            save_steps=args.save_steps,
            save_total_limit=2,
            seed=args.seed,
            report_to=[],
        ),
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["targets"],
            task_type="CAUSAL_LM",
        ),
    )
    resume = any(out.glob("checkpoint-*"))  # a Colab disconnect resumes instead of restarting
    t0 = time.perf_counter()
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(str(out / "adapter"))
    log = {
        "examples": len(ds),
        "epochs": args.epochs or s_cfg["epochs"],
        "max_seq": s_cfg["max_seq"],
        "seconds": round(time.perf_counter() - t0, 1),
        "resumed": resume,
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2) if torch.cuda.is_available() else None,
        "log_history": trainer.state.log_history,
    }
    (out / "train_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in log.items() if k != "log_history"}))


def grpo(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    from training.constraint import ScenarioConstraint
    from training.rl_rewards import TaskStore, build_rl_dataset, make_reward

    if args.arm in ("R-ADT", "R-CTV"):
        raise SystemExit(f"{args.arm} needs the twin-joint advantage (custom trainer): not implemented for the LLM yet")
    cfg = _cfg(args.config)
    store = TaskStore(Path(args.data))
    train_ids = {k for k, v in store.keys.items() if v.split == "train"}
    rows = [
        r
        for r in build_rl_dataset(Path(args.sft), store, discordance_weighting=args.arm in ("R-AD", "R-AV"))
        if r["task_id"] in train_ids
    ]
    kind = "fisher" if args.arm == "R-F" else "ctv"
    cons = ScenarioConstraint(pwr_ref={}) if args.arm == "R-AV" else None
    g = cfg["grpo"]
    lora = cfg["lora"]
    trainer = GRPOTrainer(
        model=args.init or cfg["model_id"],
        reward_funcs=[make_reward(store, kind, cons)],
        train_dataset=Dataset.from_list(rows),
        args=GRPOConfig(
            output_dir=args.out,
            learning_rate=g["lr"],
            num_generations=g["group_size"],
            per_device_train_batch_size=g["group_size"] * 2,
            max_completion_length=g["max_new_tokens"],
            max_steps=g["updates_pilot"],
            save_steps=g["checkpoint_every"],
            logging_steps=5,
            seed=args.seed,
            fp16=True,
            report_to=[],
        ),
        peft_config=LoraConfig(
            r=lora["r"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["targets"],
            task_type="CAUSAL_LM",
        ),
    )
    trainer.train()
    trainer.save_model(args.out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["sft", "grpo"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--sft", default="data/sft.jsonl")
    ap.add_argument("--init", default=None)
    ap.add_argument("--arm", default="R-A", choices=["R-F", "R-A", "R-AD", "R-AV", "R-ADT", "R-CTV"])
    ap.add_argument("--config", default="configs/t4_ctv.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=float, default=None, help="override sft.epochs")
    ap.add_argument("--save-steps", type=int, default=25, help="checkpoint interval (resume after disconnects)")
    a = ap.parse_args()
    sft(a) if a.stage == "sft" else grpo(a)


if __name__ == "__main__":
    main()
