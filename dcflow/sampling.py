"""Delayed-coupling inference with complete branchable sampler state."""

from dataclasses import dataclass, fields
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from dcflow.geometry import so3_utils
from dcflow.geometry.common.layers import sample_from
from dcflow.model import torus
from dcflow.model.torsion import torsions_mask


@dataclass
class State:
    rotations: torch.Tensor
    translations: torch.Tensor
    angles: torch.Tensor
    sequence: torch.Tensor
    carrier: torch.Tensor
    source_translations: torch.Tensor
    source_carrier: torch.Tensor

    def index(self, indices):
        return State(
            **{f.name: getattr(self, f.name)[indices].clone() for f in fields(self)}
        )


class Sampler:
    @torch.no_grad()
    def __init__(self, model, batch):
        self.model, self.batch = model, batch
        self.clean = model.encode(batch)
        self.node, self.edge = model._condition(self.clean[4], self.clean[5], batch)
        self.mask = batch["generate_mask"]

    def fixed(self, value, clean):
        mask = self.mask.reshape(*self.mask.shape, *([1] * (value.ndim - 2)))
        return torch.where(mask, value, clean)

    @torch.no_grad()
    def initialize(self):
        r, x, a, s, _, _ = self.clean
        b, l = s.shape
        rotations = torch.tensor(
            Rotation.random(b * l).as_matrix(), dtype=x.dtype, device=x.device
        ).reshape(b, l, 3, 3)
        rotations = self.fixed(rotations, r)
        noise = torch.randn_like(x)
        noise, _ = self.model.zero_center_part(noise, self.mask, self.batch["res_mask"])
        noise = self.fixed(noise, x)
        angles = self.fixed(torch.rand_like(a) * 2 * torch.pi, a)
        carrier = self.fixed(
            self.model.k * torch.randn(b, l, 20, device=x.device),
            self.model.seq_to_simplex(s),
        )
        sequence = self.fixed(sample_from(carrier.softmax(-1)), s)
        return State(
            rotations, noise, angles, sequence, carrier, noise.clone(), carrier.clone()
        )

    @torch.no_grad()
    def branch(self, state, indices):
        child = state.index(indices)
        child.source_carrier = self.fixed(
            self.model.k * torch.randn_like(child.carrier),
            self.model.seq_to_simplex(self.clean[3]),
        )
        child.carrier = child.source_carrier.clone()
        child.sequence = self.fixed(
            sample_from(child.carrier.softmax(-1)), self.clean[3]
        )
        # Chi is still inactive; fresh child sources do not change the prefix geometry.
        child.angles[..., 1:] = self.fixed(
            torch.rand_like(child.angles) * 2 * torch.pi, self.clean[2]
        )[..., 1:]
        return child

    @torch.no_grad()
    def predict(self, state, t, guide=None, step=None):
        m = self.model
        time = torch.full(
            (len(state.sequence), 1), float(t), device=state.sequence.device
        )
        st, ct = m._clock(time, m.sequence_start_t), m._clock(time, m.sidechain_start_t)
        r, x, a, logits = m.ga_encoder(
            time,
            st,
            ct,
            state.rotations,
            state.translations,
            state.angles,
            state.sequence,
            self.node,
            self.edge,
            self.mask.long(),
            self.batch["res_mask"].long(),
        )
        logits = m._sequence_logits(
            logits, state.translations, state.rotations, self.batch, st, state.angles
        )
        if guide is not None:
            logits = (
                guide(
                    step=step,
                    timestep=float(t),
                    logits=logits,
                    current_sequences=state.sequence,
                    batch=self.batch,
                )
                if step is not None
                else guide.apply_transition_bias(logits)
            )
        # Preserve exact forbidden-category support also after external feedback.
        logits = logits.masked_fill(self.batch["allowed_aa_mask"] <= 0, -1e9)
        seq = self.fixed(
            torch.distributions.Categorical(logits=logits).sample(), self.clean[3]
        )
        valid = torsions_mask.to(a.device)[seq]
        a = torch.where(valid.bool(), a, torch.zeros_like(a))
        return (
            self.fixed(r, self.clean[0]),
            self.fixed(x, self.clean[1]),
            self.fixed(a, self.clean[2]),
            seq,
        )

    @torch.no_grad()
    def advance(self, state, t, next_t, guide=None, step=None):
        r, x, a, seq = self.predict(state, t, guide, step)
        dt = float(next_t - t)
        clock = lambda time, start: max(
            0.0, min(1.0, (float(time) - start) / (1 - start))
        )
        ds = clock(next_t, self.model.sequence_start_t) - clock(
            t, self.model.sequence_start_t
        )
        dc = clock(next_t, self.model.sidechain_start_t) - clock(
            t, self.model.sidechain_start_t
        )
        xt = self.fixed(
            state.translations + (x - state.source_translations) * dt, self.clean[1]
        )
        rt = self.fixed(
            so3_utils.geodesic_t(dt * 10, r, state.rotations), self.clean[0]
        )
        da = a.new_tensor([dt, dc, dc, dc, dc])[None, None]
        at = self.fixed(torus.tor_geodesic_t(da, a, state.angles), self.clean[2])
        carrier = (
            state.carrier + (self.model.seq_to_simplex(seq) - state.source_carrier) * ds
        )
        if guide is not None:
            carrier = guide.apply_transition_bias(carrier)
        probs = carrier.softmax(-1)
        # Enforce SS identity once sequence generation is active.
        if clock(next_t, self.model.sequence_start_t) > 0:
            probs = probs * self.batch["allowed_aa_mask"]
        st = self.fixed(
            torch.distributions.Categorical(probs=probs).sample(), self.clean[3]
        )
        if clock(next_t, self.model.sidechain_start_t) > 0:
            at = torch.where(
                torsions_mask.to(at.device)[st].bool(), at, torch.zeros_like(at)
            )
            at = self.fixed(at, self.clean[2])
        return State(
            rt,
            xt,
            at,
            st,
            self.fixed(carrier, self.model.seq_to_simplex(self.clean[3])),
            state.source_translations,
            state.source_carrier,
        )

    def run(self, state, times, guide=None, step_offset=0):
        for i, (t, u) in enumerate(zip(times[:-1], times[1:])):
            state = self.advance(state, t, u, guide, i + 1 + step_offset)
        return state

    def finish(self, state, guide=None):
        r, x, a, s = self.predict(state, 1.0, guide)
        return dict(rotmats=r, trans=x, angles=a, seqs=s)


def schedule(num_steps=200, branch_time=None):
    if num_steps < 4:
        raise ValueError("At least four schedule points are required")
    times = np.linspace(0.01, 1.0, num_steps, dtype=np.float32)
    if branch_time is not None:
        # Keep the 200-point budget and place the nearest boundary exactly at tau_s.
        index = int(np.argmin(abs(times - branch_time)))
        if index in (0, len(times) - 1):
            raise ValueError("Branch boundary must be internal")
        times[index] = branch_time
    return times
