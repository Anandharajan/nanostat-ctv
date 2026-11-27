"""CPU ablation of the CTV algorithm (SPEC §8 arms) with a small proxy policy.

The proxy policy is a linear-softmax policy over a catalogue of typed-plan actions, trained with
group-normalised policy gradients (GRPO with one inner epoch, so the importance ratio is 1 and
clipping is inactive) and a KL penalty to the SFT reference. Rewards, the discordance sampler, the
twin-joint advantage and the scenario constraint are the same code paths the LLM trainer will use.

It is NOT the 1.5B language model. Results measure the effect of the training SIGNAL on a learner
of fixed capacity, and must be reported with that label.
"""
