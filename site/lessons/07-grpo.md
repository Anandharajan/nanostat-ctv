---
n: 7
title: GRPO
goal: Implement reinforcement learning with verifiable rewards, group-relative policy optimisation, in about 30 lines.
prev: 06-sft
prev_title: Supervised fine-tuning
next: 08-the-experiment
next_title: The experiment
---

## The algorithm

GRPO (introduced with DeepSeekMath and used for DeepSeek-R1) needs no value network. For each task:

1. **Sample a group** of G = 8 plans from the current policy.
2. **Score** each plan with the verifiable reward (lesson 4).
3. **Advantage** = reward minus the group mean, divided by the group standard deviation. A plan is "good" if it beats its siblings on the *same* task.
4. **Update**: raise log π of good plans and lower bad ones, with a KL penalty that keeps the policy near the SFT model.

```python
seqs  = agent.act(prompts)                                  # 8 tasks x 8 samples = 64 plans
r     = tensor([reward(task, plan) ...]).view(batch, group)
adv   = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)).view(-1)
lp    = agent.logprob(seqs)
kl    = exp(lp_ref - lp) - (lp_ref - lp) - 1                # k3 estimator of KL(policy || reference)
loss  = -(adv * lp).mean() + beta * kl.mean()
```

This is the on-policy, single-update case, so the PPO-style ratio and clipping reduce to plain policy gradient.
With several updates per batch you would add `ratio = exp(lp - lp_old)` and clip it. Try that in exercise 3.

**The budget is rollouts, not steps.** `grpo` stops after a fixed number of *generated plans* (25,600 by
default). Any comparison between rewards must give each arm the same budget; lesson 8 shows why.

## Run it

```bash
python edu/nanostat.py grpo --reward ctv       # about 1 minute on a laptop CPU
python edu/nanostat.py grpo --reward fisher
```

Measured (seed 0, from the same SFT start as lesson 6, rounded):

```
GRPO[ctv]    mean reward first/last 20 steps: 0.171 -> 0.591   eval validity 0.703  invalid_rate 0.297
GRPO[fisher] mean reward first/last 20 steps: 0.640 -> 0.843   eval validity 0.605  invalid_rate 0.395
```

Look carefully. The **Fisher-style arm has the higher reward but the lower validity.** Reward is how the
policy scores itself under *its own* reward. It is not a measure of quality, and you cannot compare it across rewards.
The question we care about is the held-out validity, which comes from the answer key and not from either reward.

One seed is an anecdote. Run with other seeds and you will see Fisher-style validity range from barely moving
to clearly moving. That is why lesson 8 uses several seeds and reports the spread.

## Exercises

1. Set `beta=0` (no KL). Does training get faster? Does the policy collapse onto one plan per world? Print the sampled plans of the last batch.
2. Change `group` from 8 to 2. The advantage is now ±1 or 0. What happens to learning speed at the same rollout budget?
3. Add PPO clipping: keep `lp_old = lp.detach()`, run 4 updates per batch with
   `ratio = exp(lp - lp_old)` and `loss = -min(ratio * adv, clip(ratio, 0.8, 1.2) * adv)`. Keep the **rollout** budget fixed. Is it more sample-efficient?
4. Plot the `log` list (the mean reward per step) for both rewards on one chart. Explain why the curves start at different heights.
