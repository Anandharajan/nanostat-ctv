---
n: 6
title: Supervised fine-tuning
goal: Teach the model by imitation, and see that it learns the teacher's habits, bad ones included.
prev: 05-tiny-gpt
prev_title: A tiny GPT
next: 07-grpo
next_title: GRPO
---

## The idea

Supervised fine-tuning (SFT) is next-token prediction on demonstrations: *(prompt, the plan a teacher wrote)*.
Real assistants are built this way before any RL. nanostat has two teachers:

- `teacher="gold"`: the registry's reference plan `[kind, A[0], alt]`.
- `teacher="textbook"` (the default): a "textbook analyst" who always uses the classic parametric test
  (Student t, paired t, Pearson), **whatever the data profile says**. It is a realistic and common habit.

The experiment starts from the textbook teacher on purpose. The question for RL is then concrete: *can the reward
make the model un-learn a bad habit?*

```python
def sft(agent, tasks, steps=150, lr=3e-3, batch=32, seed=0, teacher="textbook", smoothing=0.3):
    ...
    loss = sum(F.cross_entropy(lp, t, label_smoothing=smoothing) for lp, t in agent.slot_logprobs(b)) / 3
```

**Label smoothing 0.3** keeps some probability on the other words in each slot. Without it SFT makes the policy
nearly deterministic, and RL then has nothing to explore. If every sample is the same plan, every advantage in
lesson 7 is zero.

## Run it

```bash
python edu/nanostat.py sft      # 40 demonstrations, 150 steps, a few seconds
```

```
SFT (textbook teacher) final loss 0.77 | eval {'validity': 0.468, 'invalid_rate': 0.532, 'tail_acc': 1.0, 'agent_FPR': 0.083}
```

(Rounded. The loss stays well above 0 because of the smoothing.) The model learned the tail from the question words
perfectly (`tail_acc` 1.0). It also learned the habit: on 53% of test tasks it picks a method the registry marks
**invalid**. Its validity, 47%, is exactly the share of tasks where the textbook test happens to be admissible.
`agent_FPR` here comes from only the 219 null tasks in this test set, so it is noisy. Lesson 8 uses 700 null tasks.

## Exercises

1. Train with `teacher="gold"`. What validity do you get? Why isn't it 100%? (Hint: the data profile is noisy; see lesson 3, exercise 1.)
2. Set `smoothing=0.0` and print the probability of the most likely method on a few prompts. Then run lesson 7
   on top of it and watch the GRPO reward curve stay flat.
3. Use 400 demonstrations instead of 40. Does a *stronger* habit make RL's job harder?

> **Why this matters.** SFT copies; it does not judge. If the demonstrations come from people, or from a
> bigger model, with systematic mistakes, the student copies those too. RL with a well-designed verifiable reward is one way to correct that afterwards.
