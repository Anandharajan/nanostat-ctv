# nanostat: the teaching track

`nanostat.py` is the smallest complete version of NanoStat-CTV in a single file of about 580 lines. It has hypothesis tests written from scratch, a task world with answer keys, typed plans with a verifier, two verifiable rewards, a 104k-parameter GPT, SFT, GRPO and a compute-matched experiment.

- **Course site and in-browser demo:** https://anandharajan.github.io/nanostat-ctv/ (source in `site/`)
- **Colab:** [`nanostat_colab.ipynb`](https://colab.research.google.com/github/Anandharajan/nanostat-ctv/blob/main/edu/nanostat_colab.ipynb)

```bash
pip install numpy scipy torch          # CPU torch is fine
python edu/nanostat.py stats           # lesson 1  ... one sub-command per lesson
python edu/nanostat.py compare         # lesson 8: the full experiment, ~6 min on a laptop CPU
```

| Lesson | Command |
|---|---|
| 1 Tests from scratch | `stats` |
| 2 Simulate the null | `null` |
| 3 Tasks, answer keys, typed plans | `tasks` |
| 4 Rewards and reward hacking | `rewards` |
| 5 A tiny GPT | `model` |
| 6 Supervised fine-tuning | `sft` |
| 7 GRPO | `grpo --reward ctv` / `grpo --reward fisher` |
| 8 The experiment | `compare` |
| 9 From nano to real | (reading) |

Measured outputs are in `results/edu/`. The 3-seed result (`compare_3seeds.txt`, 25,600 plans per arm) is below:

| Arm | Validity | Invalid-method rate |
|---|---|---|
| SFT only | 0.436 ± 0.024 | 0.532 ± 0.000 |
| GRPO, Fisher-style reward | 0.462 ± 0.009 | 0.532 ± 0.000 |
| GRPO, CTV reward | 0.688 ± 0.118 | 0.267 ± 0.107 |

This is a toy world built for teaching. Its numbers say nothing about LLM performance or about Fisher-R1 itself; lesson 8 explains what they do and do not support. The file is self-contained by design and does not import the `nanostat` package.
