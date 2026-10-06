# Inference protocol and paper correspondence

## Generator: Sections 3.1–3.2, Appendix C

The state consists of residue translations, rotations, five angles (AF-style ψ
and χ1–χ4), a 20-channel sequence carrier, and its sampled categorical sequence.
The receptor is fixed. Node/pair widths are 128/64; the shared trunk has six
IPA–Transformer blocks. Configuration is in `config/model.json`.

The local clocks are `s_seq=clip((t-0.5)/0.5,0,1)` and
`s_chi=clip((t-0.8)/0.2,0,1)`. Inactive sequence inputs use UNK and inactive χ
inputs use zero-angle encoding. Backbone and ψ continue throughout the trajectory.
The current noisy backbone and ψ, together with observed receptor N/CA/C/O,
condition the frozen ProteinMPNN. Centered log-probabilities are added to native
sequence logits with the learned scalar gate. SS endpoint identities use Cys-only
support. Frames, sequence and angles are reconstructed into atom14 heavy atoms.

The sampler follows the endpoint-prediction update convention:

```text
x_next = x + (x_endpoint - x_source) * dt
R_next = SO3Geodesic(R, R_endpoint, 10*dt)
angles_next = TorusGeodesic(angles, angles_endpoint, [dt,dchi,dchi,dchi,dchi])
u_next = u + (Encode(sequence_endpoint) - u_source) * dseq
```

Categorical draws occur during the trajectory and at the final prediction.
`Encode(a)=10*onehot(a)-5`. The final output uses a separate endpoint prediction
at `t=1`. Defaults use 200 points, hence 199 updates and one final prediction.

## Guide: Appendix C.6 and E

Events occur at loop steps 125, 150, 175 and 199. At each event, three maximum-
entropy positions per candidate each contribute up to three admissible single-
residue alternatives. Model 1 multimer v3 uses one recycle, the receptor MSA,
cyclic offset type 2 and the specified hotspots. Only sequences are transmitted.

The online score is `S=ipTM+pLDDT_peptide/100+1-hotspot_iPAE_normalized`.
With `e_base=abs(3-S_base)` and `e_mut=abs(3-S_mut)`, the tested category receives:

```text
50*t*(e_base-e_mut)/max(e_base,0.1)          if e_mut < e_base
-0.3*50*t*(e_mut-e_base)/max(e_mut,0.1)     otherwise
```

Historical bias decays by 0.9 per evaluation event. Bias affects later logits,
the updated carrier, and the final prediction. Carrier addition is not multiplied
by a clock increment. No AF coordinates replace the generative state.

## Cascaded search: Section 3.3, Appendix B.4

1. Generate `K` prefixes. The cascade keeps the 200-point budget and places the
   grid point nearest the sequence boundary exactly at `t=0.5`.
2. Select `K0` roots with the structural potential documented in the README.
   The declared distance is **shape RMSD** of ordered Cα coordinates after proper
   Kabsch alignment. Residue order is fixed; no mirror or cyclic permutation fit
   is used. Parameters are σ=1.75 Å, T=2.5, ε=1e-8.
3. Create `M` children per root. The complete continuation state includes current
   rotation, translation, ψ/χ, and the original translation source. Fresh
   sequence carriers replace unused parent sequence sources, and inactive χ is
   independently initialized. Backbone updates continue after branching.
4. Apply the same online guide during child continuations.
5. Run five initial refinement updates per generated leaf, with ipTM objective.
6. Compute `Q=0.75*(1-hotspot_iPAE_normalized)+0.25*ipTM`. Draw `N0` distinct
   leaves sequentially, with positive softmax weights `exp(Q/T_Q)` and `T_Q=0.1`.
7. Restore each survivor's optimizer state, sequence parameters, inputs and PRNG
   state. Continue the refinement schedule with ipTM and hotspot-iPAE objectives.
8. In the temperature-annealed soft stage, stop after five consecutive updates
   without strict improvement in the reported hotspot composite. Always execute
   the hard completion stage. Return fixed-sequence cofolded structures.

Selection temperatures and the without-replacement policy are recorded in
configuration and output metadata.

## Refinement: 160-update preset

The supplied preset allocates 100/50/10 updates to logits-to-soft, annealed soft,
and hard stages. The sequence optimizer is ColabDesign's normalized-gradient SGD
with learning rate 0.1. Dropout is enabled in the first two stages and disabled
in the hard stage. Model 1, one recycle is used throughout.

The differentiable callback objectives are `(1-ipTM)` and normalized symmetric
hotspot interface PAE, with weights 1 and 1. During the five screening updates,
the hotspot-PAE weight is zero. Other inherited ColabDesign objective weights
are zeroed explicitly. Plain refinement executes all 160 updates; in a cascade,
the five screening updates count toward the same cap and survivors resume at
update six. Early stopping applies to the annealed soft stage. The stopping
composite is `ipTM+pLDDT_peptide/100+1-hotspot_iPAE_normalized`, separate from Q.

## Independent evaluation

`evaluate.py` uses all five multimer-v3 models, three recycles, no dropout,
receptor-only conditioning, cyclic offset type 2, and final discrete sequences.
PAE is symmetrized across interface directions and divided by 31. Candidate
metrics are five-model means. The interface composite uses all receptor residues;
the hotspot composite uses the specified hotspot subset.

Raw generation, online guide, refinement and independent evaluation remain
separately identified in outputs, so their numerical protocols are recoverable.

## Package contents

The inference package includes generator tensors, external-model interfaces,
example inputs, numerical settings and executable workflow checks.
