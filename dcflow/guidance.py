"""Persistent single-substitution AFCyc feedback (Appendix E)."""

import torch
from dcflow.geometry.protein.constants import resindex_to_ressymb


class AFCycGuidance:
    def __init__(
        self,
        evaluator,
        mask,
        steps=(125, 150, 175, 199),
        scale=50.0,
        positions=3,
        alternatives=3,
    ):
        self.evaluator, self.mask = evaluator, mask
        self.steps = set(steps)
        self.scale, self.positions, self.alternatives = scale, positions, alternatives
        self.accumulator = None
        self.events = []

    def apply_transition_bias(self, values):
        return values if self.accumulator is None else values + self.accumulator

    def __call__(self, step, timestep, logits, current_sequences, batch):
        if step not in self.steps:
            return self.apply_transition_bias(logits)
        probabilities = logits[:, self.mask].softmax(-1)
        seq = current_sequences[:, self.mask]
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
        positions = entropy.topk(min(self.positions, seq.shape[1]), dim=-1).indices
        mutants, records = [], []
        allowed = batch["allowed_aa_mask"][:, self.mask]
        for b, sites in enumerate(positions.tolist()):
            for i in sites:
                alternatives = [
                    a
                    for a in probabilities[b, i].argsort(descending=True).tolist()
                    if a != int(seq[b, i]) and allowed[b, i, a] > 0
                ][: self.alternatives]
                for a in alternatives:
                    mutant = seq[b].clone()
                    mutant[i] = a
                    mutants.append(mutant)
                    records.append((b, i, a))
        decode = lambda s: "".join(resindex_to_ressymb[int(a)] for a in s)
        results = self.evaluator.evaluate(
            [decode(s) for s in seq] + [decode(s) for s in mutants]
        )
        scores = [
            r["ipTM"] + r["peptide_mean_pLDDT"] / 100 + 1 - r["hotspot_iPAE_normalized"]
            for r in results
        ]
        if self.accumulator is None:
            self.accumulator = torch.zeros_like(logits)
        self.accumulator.mul_(0.9)
        indices = self.mask.nonzero().flatten()
        rewards = []
        for (b, i, a), score in zip(records, scores[len(seq) :]):
            base, mut = abs(3 - scores[b]), abs(3 - score)
            value = (
                self.scale
                * timestep
                * (
                    (base - mut) / max(base, 0.1)
                    if mut < base
                    else -0.3 * (mut - base) / max(mut, 0.1)
                )
            )
            self.accumulator[b, indices[i], a] += value
            rewards.append(value)
        self.events.append(
            dict(
                step=step,
                time=timestep,
                base_scores=scores[: len(seq)],
                mutant_scores=scores[len(seq) :],
                mutations=records,
                rewards=rewards,
            )
        )
        return self.apply_transition_bias(logits)
