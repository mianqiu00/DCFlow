# External models and parameters

DCFlow weights are included under `checkpoints/`. ProteinMPNN and AlphaFold
parameters remain external and are obtained from their public distributions.

## ProteinMPNN

Repository: <https://github.com/dauparas/ProteinMPNN>

```bash
git clone https://github.com/dauparas/ProteinMPNN.git
git -C ProteinMPNN checkout 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57
export PROTEINMPNN_SOURCE="$(realpath ProteinMPNN)"
export PROTEINMPNN_WEIGHTS="$PROTEINMPNN_SOURCE/vanilla_model_weights/v_48_020.pt"
```

The repository includes the required `vanilla_model_weights/v_48_020.pt`.
Its SHA256 is:

```text
c9cb4a671d79604111231f8dbfc7c590e06f1197453b7a6854ac6661a642f5bd
```

The adapter imports `protein_mpnn_utils.py` from that directory and loads the
checkpoint strictly. It uses all-backbone N/CA/C/O inputs, 48 neighbors, three
encoder/decoder layers, no coordinate augmentation and no dropout. ProteinMPNN
weights remain frozen; the learned fusion gate is part of the included DCFlow
checkpoint. The first 20 output channels are the shared standard-AA vocabulary
`ACDEFGHIKLMNPQRSTVWY`.

You may also pass `--mpnn-source` and `--mpnn-weights` to `inference.py`.

## AfCycDesign interface via ColabDesign

The supplied adapter implements the AfCycDesign cyclic-offset interface on the
public ColabDesign backend:

- <https://github.com/sokrypton/ColabDesign>
- Tested commit: `e31a56fe1d9b4de25c8697f3a28b75892941cc72` (version 1.1.3).

Install the separate `environment-afcyc.yaml`. The core PyTorch package does not
need to be installed in this JAX environment: the launcher adds the release root
to the worker's import path, and the worker imports only backend adapter modules.

For an existing folding environment, the equivalent external source setup is:

```bash
git clone https://github.com/sokrypton/ColabDesign.git
git -C ColabDesign checkout e31a56fe1d9b4de25c8697f3a28b75892941cc72
export COLABDESIGN_SOURCE="$(realpath ColabDesign)"
export FOLDING_PYTHON=/path/to/folding/environment/bin/python
```

Install the dependencies from `environment-afcyc.yaml` in that environment.
The tested public revision already accepts an `offset` matrix in multimer
relative-position encoding; no private fork is required. The release supplies
the cyclic offset, MSA projection, multichain handling, metrics, differentiable
refinement objectives and subprocess protocol.

## AlphaFold-Multimer-v3 parameters

Obtain the official parameter archive through the AlphaFold instructions:

- <https://github.com/google-deepmind/alphafold#model-parameters>
- Official v3-era archive:
  <https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar>

Extract the archive into an external directory and set:

```bash
export ALPHAFOLD_PARAMS=/path/to/alphafold/params
```

The directory must contain `params_model_1_multimer_v3.npz` through
`params_model_5_multimer_v3.npz` for five-model final evaluation. Guide and
refinement use model 1. The worker's startup fails explicitly if the configured
backend cannot load its parameters.

Follow the upstream parameter license/terms. These files are not redistributed
inside the anonymous supplement.

## GPU memory and environments

The measured backend smoke used Python 3.12, JAX 0.9.1, Haiku 0.0.16 and Optax
0.2.8 with NVIDIA CUDA support. `environment-afcyc.yaml` pins these backend
versions. The generator environment pins the tested PyTorch 2.8.0 release
(the measured installation used its CUDA 12.8 build). Use a compatible NVIDIA
driver, or install the appropriate PyTorch wheel for your platform before
`pip install -e .`.

Set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. `--oracle-gpu` controls the folding
worker's `CUDA_VISIBLE_DEVICES`; it does not change the generator device. A
receptor pocket limits the generator's dense residue-pair memory, while folding
uses the configured full receptor and MSA. CPU tests are supported for the core
model, but GPU execution is recommended for cofolding.
