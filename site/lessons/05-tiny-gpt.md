---
n: 5
title: A tiny GPT
goal: Read and run a complete decoder-only transformer, about 40 lines, the same architecture family as GPT-2 and nanoGPT.
prev: 04-rewards
prev_title: Rewards and reward hacking
next: 06-sft
next_title: Supervised fine-tuning
---

## The model

Section 5 of `edu/nanostat.py` is a GPT in miniature. If you have read
[nanoGPT](https://github.com/karpathy/nanoGPT)'s `model.py`, every line will look familiar:

```python
class SelfAttention(nn.Module):  # multi-head causal self-attention
    ...
    att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.h)
    att = att.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))  # causal mask
    return self.proj((F.softmax(att, dim=-1) @ v).transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):  # pre-LayerNorm residual block
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):  # token + position embeddings, blocks, LayerNorm, tied output head
    ...
    self.head.weight = self.tok.weight  # weight tying, as in GPT-2
```

| | nanostat | GPT-2 small |
|---|---|---|
| embedding size | 64 | 768 |
| heads / layers | 4 / 2 | 12 / 12 |
| context | 19 tokens | 1,024 |
| vocabulary | 43 words | 50,257 BPE tokens |
| parameters | **104,064** | ~124 M |

The vocabulary is whole words (`build_vocab`). There is no BPE, because the observations use a small fixed language.

## Prompt in, plan out

`Agent` wraps the model with a tokenizer and a **grammar**:

- `encode(obs)` left-pads the observation to 16 tokens and ends it with `<plan>`, so the plan always starts at the same position.
- `act(prompts)` generates 3 tokens. Slot 1 may only be a design, slot 2 a method, slot 3 a tail (`SLOTS`).
  Each step masks the logits to the allowed words, then samples, or takes the argmax when `greedy=True`.
- `logprob(seqs)` returns log π(plan | prompt), the sum of the 3 slot log-probabilities. Lessons 6 and 7 train with it.

Constrained decoding is how real agent systems force valid JSON or tool calls. Here it means every output is
*shaped* like a plan. It can still be the *wrong* plan: `['two_group', 'spearman', ...]` passes the grammar,
but `execute` rejects it because Spearman is not a two-group test.

## Run it

```bash
python edu/nanostat.py model
```

```
parameters: 104064
untrained plans: [['two_group', 'spearman', 'greater'], ['two_group', 'spearman', 'greater'], ['two_group', 'spearman', 'greater']]
```

The untrained model has no idea yet: its plan cannot even run.

## Exercises

1. Remove the causal mask. Training still runs (try lesson 6). Why does it not matter much *here*, and why would it be fatal for a language model generating text?
2. Remove the grammar from `act` and sample from the full vocabulary. How often is an untrained plan even three plan words?
3. Count the parameters by hand: embeddings, then per block (qkv, proj, 2 LayerNorms, MLP), then the final
   LayerNorm. Remember the head is tied, so it adds nothing. Check your total against 104,064.
