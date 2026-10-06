# DCFlow — inference release

**Target-conditioned cyclic peptide design via delayed-coupling flow matching.**

This package provides the trained DCFlow generator and its inference pipeline:

- Delayed sequence and side-chain activation, with continuous backbone evolution.
- Topology-conditioned head-to-tail (HT) and terminal-disulfide (SS) generation.
- A frozen ProteinMPNN prior fused into sequence predictions throughout generation.
- In-trajectory AfCycDesign guidance and sequence-initialized cofolding refinement.
- Cascaded search: structural root selection, independent sequence continuations,
  five-update quality screening, and early-stopped refinement.

The included checkpoint contains **original FP32 DCFlow tensors**, including the
learned prior gate, topology adapters, and modality-clock projection. External
ProteinMPNN and AlphaFold weights are downloaded separately. No training dataset,
training checkpoint, or original development directory is required.

## 1. Installation

Linux and an NVIDIA GPU are recommended. For the generator:

```bash
conda env create -f environment.yaml
conda activate dcflow
```

Alternatively, install a CUDA-compatible PyTorch and run `pip install -e .`.
Run the commands below from the extracted `DCFlow/` directory. This editable
installation also supports importing `dcflow` from another working directory.

Configure ProteinMPNN as described in **[EXTERNAL_MODELS.md](EXTERNAL_MODELS.md)**:

```bash
export PROTEINMPNN_SOURCE=/path/to/ProteinMPNN
export PROTEINMPNN_WEIGHTS="$PROTEINMPNN_SOURCE/vanilla_model_weights/v_48_020.pt"
```

Direct generation needs only the generator environment and ProteinMPNN. The
guided/refinement modes additionally use a separate JAX environment:

```bash
conda env create -f environment-afcyc.yaml
export FOLDING_PYTHON="$(conda run -n dcflow-afcyc python -c 'import sys; print(sys.executable)')"
export ALPHAFOLD_PARAMS=/path/to/alphafold/params
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

The optional backend is a version-pinned public ColabDesign installation, with
the cyclic-offset, receptor-MSA, guidance and refinement interfaces supplied
here. See the external-model guide for download links and GPU configuration.

## 2. Direct full-atom generation

```bash
python inference.py examples/5LSO.json --output results/direct --num-designs 32
```

Default generation uses 200 time points and seed 2026. The same command supports
terminal SS generation by setting `"topology": "SS"` in the input JSON. The
terminal positions are constrained to cysteine during active sequence generation.

### Input JSON

The included example contains a public 5LSO receptor pocket, full receptor and
receptor MSA. Paths resolve relative to the input JSON, independent of the current
working directory:

```json
{
  "case_id": "my_target",
  "pocket": "pocket.pdb",
  "receptor": "receptor.pdb",
  "receptor_chains": "A",
  "length": 12,
  "center": [10.0, 20.0, 30.0],
  "topology": "HT",
  "receptor_msa": "receptor.a3m",
  "hotspots": "A25,A40,A60"
}
```

`center` is in the receptor's coordinate frame, in Å. Instead of `length` and
`center`, provide `peptide_reference`; only its length and Cα center define the
generation region. The generated region's input sequence and geometry are masked
inside the conditional encoder. Pocket and full-receptor coordinates must share
the same frame. PDB backbone N/CA/C atoms are required for retained residues.

For direct generation, `receptor_msa` and `hotspots` are unnecessary; `receptor`
can be omitted to write a complex with the pocket. For cofolding, supply all three
and use `HT`. Hotspot selectors refer to the numbering in `receptor`, with chain
IDs preserved. Multiple receptor chains use, for example, `"A,B,C"`. The MSA
query must map to the corresponding receptor sequence; its columns are projected
onto the retained receptor residues by the supplied adapter.

## 3. In-trajectory guide and refinement

```bash
# Four online guidance events during the DCFlow trajectory.
python inference.py examples/5LSO.json --mode guided --output results/guided

# Guided generation followed by 100/50/10 AfCycDesign refinement updates.
python inference.py examples/5LSO.json --mode refine --output results/refined

# Sequence-initialized refinement without trajectory guidance.
python inference.py examples/5LSO.json --mode refine --no-guidance --output results/refined_unguided
```

Online guidance uses model 1 of AlphaFold-Multimer-v3, one recycle, cyclic offset
type 2, and the hotspot score. Each event evaluates the current sequence and
single-residue alternatives, then retains the resulting sequence bias for later
flow steps. Refinement initializes from generated sequences and uses a
100-update logits-to-soft stage, 50-update temperature-annealing stage, and
10-update hard stage, totaling 160 updates.

The worker is a persistent subprocess. To place it on another GPU, use
`--oracle-gpu 1`; the generator's device is selected with `--device cuda:0`.
Without this option both inherit the same visible GPUs. Limit the generation
batch or use separate devices when GPU memory is constrained.

## 4. Cascaded search

```bash
python inference.py examples/5LSO.json --mode cascade --output results/search
```

The benchmark preset in `config/inference.json` uses:

| Setting | Value |
|---|---:|
| Initial backbone prefixes `K` | 16 |
| Retained roots `K0` | 8 |
| Independent sequence children per root `M` | 8 |
| Generated leaves `N = K0 × M` | 64 |
| Retained refinement candidates `N0` | 32 |
| Structural Gaussian width | 1.75 Å |
| Structural selection temperature | 2.5 |
| Structural potential epsilon | 1e-8 |
| Quality-selection temperature | 0.1 |
| Screening updates | 5 |
| Soft-stage non-improvement patience | 5 |

At `t=0.5`, root selection uses ordered-Cα RMSD after proper Kabsch alignment.
Selection is without replacement: the first root is uniform, and subsequent
probabilities are proportional to
`(epsilon + mean(exp(-RMSD²/(2*sigma²))))**(-1/temperature)`.
Each retained root supplies its complete sampler memory to independent children,
including the original translation source. Each child receives a fresh sequence
source, and the backbone remains flexible.

The leaves receive five ipTM-targeted refinement updates. Survivors are drawn
without replacement with weights proportional to `exp(Q / 0.1)`, where
`Q = 0.75*(1-hotspot_iPAE_normalized) + 0.25*ipTM`. Their optimizer states and
random keys resume from screening. Refinement stops its soft stage after five
updates without improvement in the hotspot composite, then completes the hard
stage to return discrete sequences.

Edit a copy of `config/inference.json` and pass `--settings my_settings.json` to
change budgets. The TL1A budget described in the paper is `K=8,K0=4,M=6,N0=12`.
All numerical choices are recorded in the output. Detailed algorithm mapping is
in **[PROTOCOL.md](PROTOCOL.md)**.

## 5. Independent final evaluation

```bash
python evaluate.py examples/5LSO.json --results results/search --output results/search_eval
```

This evaluates final discrete sequences with five AlphaFold-Multimer-v3 models,
three recycles, receptor-only conditioning, and no binder template or initial
pose. `metrics.json` contains per-model results and candidate-wise means.
Strict hits use mean ipTM ≥ 0.7, peptide pLDDT ≥ 70, and normalized interface
iPAE ≤ 0.3. The online guide and final evaluation have separate budgets and
outputs. Neural PDB outputs precede the paper's optional physical relaxation.

## 6. Outputs

Use a new or empty output directory for each run:

```text
results/search/
├── case.json                  resolved input specification
├── settings.json              complete numerical configuration
├── generated/
│   ├── candidate_000.pdb       receptor + generated peptide, original frame
│   ├── candidate_000_peptide.pdb
│   └── candidates.json        sequences, topology, chain IDs, relative paths
├── refined/                   when refinement is enabled
│   ├── candidate_000.pdb
│   └── candidate_000.json     sequence, scores, update trace, stopping status
└── summary.json               roots, branching, guide events, screening, outputs
```

`summary.json` retains the mapping from generated leaves to selected refinement
candidates. Peptide chain P is used when available; output metadata records the
actual chain. Input receptor chain identities and coordinates are retained in
generated complexes.

## 7. Verification

```bash
python -m unittest discover -s tests -v
python smoke_test.py --output /tmp/dcflow-smoke
python smoke_test.py --with-afcyc --oracle-gpu 1 --output /tmp/dcflow-af-smoke
```

Smoke tests use the released DCFlow weights and real external ProteinMPNN. The
last command additionally exercises real AF scoring, guidance, screening,
optimizer-state continuation and refinement. The compact smoke settings reduce
candidate budgets and trajectory length; production defaults remain in
`config/inference.json`. Measured validation is documented in
**[SMOKE_TEST.md](SMOKE_TEST.md)**.

## Source map

| Component | File |
|---|---|
| Trained generator | `dcflow/model/network.py` |
| Shared geometric trunk and clocks | `dcflow/model/ga.py` |
| Topology conditioning | `dcflow/model/cyclic.py` |
| External ProteinMPNN integration | `dcflow/model/mpnn_prior.py` |
| Joint trajectory and branchable state | `dcflow/sampling.py` |
| Structural / quality selection | `dcflow/steering.py` |
| Persistent cofolding feedback | `dcflow/guidance.py` |
| External AFCyc worker / refinement | `dcflow/adapters/afcyc_worker.py` |

This is an inference release. Third-party component attribution and license
notices are provided in `THIRD_PARTY_NOTICES.md` and `licenses/`.
