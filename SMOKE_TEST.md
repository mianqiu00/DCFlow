# Verification record

The release was exercised with real DCFlow tensors and external model weights.
The following checks passed during package preparation:

| Check | Executed setting | Result |
|---|---|---|
| Core contract tests | Delayed input visibility, clocks, structural selector, branch source preservation, fixed receptor, split continuation equivalence, persistent guide bias | 6 passed |
| HT and terminal SS smoke | 2 candidates each, 12 trajectory points, real ProteinMPNN | Passed; SS terminal Cys verified |
| Complete cascade smoke | K=3, K0=1, M=2, N0=1; real AF guide; 5 screening updates; optimizer continuation; 5/3/1 cap | Passed; 2 generated, 1 refined; early-stop branch exercised |
| Default generation schedule | 2 candidates, 200 points | Passed |
| Full refinement cap | 1 generated seed, 100/50/10 stages | Passed; exactly 160 optimization updates |
| Independent evaluation | 2 generated sequences × 5 multimer-v3 models × 3 recycles | Passed; 10 structures and confidence records |
| Public external backend | Fresh public ProteinMPNN and ColabDesign checkouts at documented revisions | Passed |

The complete real-backend compact smoke took approximately **161 seconds**.
One 160-update refinement took approximately **277 seconds**, including backend
startup/compilation. Timings are hardware and cache dependent. The generator and
JAX worker were assigned separate GPUs for the cascade check.

Measured software: generator Python 3.9, PyTorch 2.8.0+cu128, NumPy 1.26.4,
SciPy 1.13.1, Biopython 1.85, einops 0.8.2; folding Python 3.12, JAX 0.9.1,
Haiku 0.0.16, Optax 0.2.8. External sources and weight identifiers are recorded
in `EXTERNAL_MODELS.md`. The supplied environments pin these principal versions.

## Repeat the checks

After installation and configuring external model locations:

```bash
python -m unittest discover -s tests -v
python smoke_test.py --output /tmp/dcflow-core-check
python smoke_test.py --with-afcyc --oracle-gpu 1 --output /tmp/dcflow-full-check
```

Use new output directories. The full smoke writes its explicit compact settings
and a machine-readable `smoke_report.json`. It uses actual model parameters;
it does not substitute a mock oracle. For a single GPU, omit `--oracle-gpu` and
ensure sufficient free memory for both frameworks.

The tests check execution, input/output contracts, and sampler state handling.
Production budgets are supplied in `config/inference.json`.
