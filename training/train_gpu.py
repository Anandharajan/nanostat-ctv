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


def sft(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    cfg = _cfg(args.config)
    rows = [json.loads(x) for x in Path(args.sft).read_text(encoding="utf-8").splitlines() if x.strip()]
    rows = rows[: cfg["sft"]["examples"]]
    ds = Dataset.from_list([{"prompt": r["prompt"], "completion": r["completion"]} for r in rows])
    lora = cfg["lora"]
    trainer = SFTTrainer(
        model=cfg["model_id"],
        train_dataset=ds,
        args=SFTConfig(
            output_dir=args.out,
            per_device_train_batch_size=cfg["sft"]["micro_batch"],
            gradient_accumulation_steps=cfg["sft"]["grad_accum"],
            learning_rate=cfg["sft"]["lr"],
            num_train_epochs=cfg["sft"]["epochs"],
            max_length=cfg["sft"]["max_seq"],
            fp16=True,
            logging_steps=10,
            save_steps=100,
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
    trainer.train()
    trainer.save_model(args.out)


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
    a = ap.parse_args()
    sft(a) if a.stage == "sft" else grpo(a)


if __name__ == "__main__":
    main()
