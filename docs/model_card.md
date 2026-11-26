# Model card — NanoStat-CTV-1.5B

**Status: no model has been trained. NOT RUN.**

The planned model is a QLoRA adapter (r = 8) on `Qwen/Qwen2.5-Coder-1.5B-Instruct`, trained with programmatic SFT and then CTV-GRPO on a single T4 (see `configs/t4_ctv.yaml`, SPEC §7–8). This card will be filled with measured results, compute, seeds and failure analysis only after milestones M2–M7 have actually been run.

v0.1 ships the deterministic rule planner (arm B0) as the only planner.
