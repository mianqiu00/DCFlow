import copy
import json
from pathlib import Path
import unittest
import numpy as np
import torch

from dcflow.model.network import DCFlow
from dcflow.model.ga import GAEncoder
from dcflow.runtime import config, load_model, seed_all
from dcflow.steering import select, quality_select
from dcflow.sampling import Sampler, schedule
from dcflow.inputs import read_case, prepare, make_batch
from dcflow.guidance import AFCycGuidance

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def test_delayed_input_visibility(self):
        cfg = config(json.loads((ROOT / "config/model.json").read_text()))
        encoder = GAEncoder(cfg.encoder.ipa).eval()
        n = 4
        r = torch.eye(3).expand(1, n, 3, 3)
        x = torch.randn(1, n, 3)
        a = torch.randn(1, n, 5)
        a2 = a.clone()
        a2[..., 1:] += 2
        s = torch.zeros(1, n, dtype=torch.long)
        h = torch.randn(1, n, 128)
        z = torch.randn(1, n, n, 64)
        mask = torch.ones(1, n, dtype=torch.long)
        t = torch.tensor([[0.25]])
        zero = torch.zeros_like(t)
        with torch.no_grad():
            one = encoder(t, zero, zero, r, x, a, s, h, z, mask, mask)
            two = encoder(t, zero, zero, r, x, a2, s + 5, h, z, mask, mask)
        for v, w in zip(one, two):
            torch.testing.assert_close(v, w, rtol=0, atol=0)

    def test_clocks_and_boundary(self):
        t = torch.tensor([[0.25], [0.5], [0.75], [1.0]])
        torch.testing.assert_close(
            DCFlow._clock(t, 0.5), torch.tensor([[0.0], [0.0], [0.5], [1.0]])
        )
        times = schedule(200, 0.5)
        self.assertEqual(len(times), 200)
        self.assertIn(0.5, times)
        self.assertTrue(np.all(np.diff(times) > 0))

    def test_fks_rigid_invariance_and_budget(self):
        rng = np.random.default_rng(3)
        xyz = rng.normal(size=(8, 6, 3))
        one = select(xyz, 4, 42)
        two = select(xyz + np.array([10, 20, 30]), 4, 42)
        self.assertEqual(one, two)
        self.assertEqual(len(set(one)), 4)
        self.assertEqual(len(set(quality_select([0.0, 0.1, 0.8], 2, 3))), 2)

    def test_branch_memory_and_fixed_receptor(self):
        # No external weights needed for this structural sampler contract test.
        cfg = config(json.loads((ROOT / "config/model.json").read_text()))
        cfg.mpnn_prior.enabled = False
        model = DCFlow(cfg).eval()
        case = read_case(ROOT / "examples/5LSO.json")
        item, _ = prepare(case)
        parent = Sampler(model, make_batch(item, 2, "cpu"))
        state = parent.initialize()
        child = Sampler(model, make_batch(item, 4, "cpu"))
        branched = child.branch(state, [0, 0, 1, 1])
        torch.testing.assert_close(
            branched.source_translations[0], state.source_translations[0]
        )
        torch.testing.assert_close(branched.translations[0], branched.translations[1])
        self.assertFalse(
            torch.equal(branched.source_carrier[0], branched.source_carrier[1])
        )
        after = child.advance(branched, 0.5, 0.55)
        fixed = ~child.mask
        torch.testing.assert_close(
            after.translations[fixed], child.clean[1][fixed], rtol=0, atol=0
        )
        torch.testing.assert_close(
            after.rotations[fixed], child.clean[0][fixed], rtol=0, atol=0
        )

    def test_split_continuation_matches_uninterrupted(self):
        cfg = config(json.loads((ROOT / "config/model.json").read_text()))
        cfg.mpnn_prior.enabled = False
        model = DCFlow(cfg).eval()
        item, _ = prepare(read_case(ROOT / "examples/5LSO.json"))
        sampler = Sampler(model, make_batch(item, 1, "cpu"))
        seed_all(81)
        state = sampler.initialize()
        times = schedule(6, 0.5)
        split = int(np.argmin(abs(times - 0.5)))
        rng = torch.get_rng_state()
        full = sampler.run(copy.deepcopy(state), times)
        torch.set_rng_state(rng)
        prefix = sampler.run(copy.deepcopy(state), times[: split + 1])
        resumed = sampler.run(prefix, times[split:], step_offset=split)
        for key in full.__dataclass_fields__:
            torch.testing.assert_close(
                getattr(full, key), getattr(resumed, key), rtol=0, atol=0
            )

    def test_guidance_persists_and_decays_only_at_events(self):
        class Oracle:
            def evaluate(self, sequences):
                return [
                    dict(
                        ipTM=0.2 if i == 0 else 0.8,
                        peptide_mean_pLDDT=80.0,
                        hotspot_iPAE_normalized=0.2,
                    )
                    for i in range(len(sequences))
                ]

        guide = AFCycGuidance(
            Oracle(), torch.tensor([True, True]), steps=[2], positions=1, alternatives=1
        )
        logits = torch.zeros(1, 2, 20)
        batch = dict(allowed_aa_mask=torch.ones_like(logits))
        seq = torch.zeros(1, 2, dtype=torch.long)
        guide(step=2, timestep=0.65, logits=logits, current_sequences=seq, batch=batch)
        stored = guide.accumulator.clone()
        self.assertGreater(stored.sum().item(), 0)
        torch.testing.assert_close(guide.apply_transition_bias(logits), stored)
        for step in (3, 4):
            torch.testing.assert_close(
                guide(
                    step=step,
                    timestep=0.7,
                    logits=logits,
                    current_sequences=seq,
                    batch=batch,
                ),
                stored,
            )
        self.assertEqual(len(guide.events), 1)


if __name__ == "__main__":
    unittest.main()
