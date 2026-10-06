# Third-party notices

DCFlow combines delayed-coupling generation, topology conditioning, a frozen
inverse-folding prior, and cascaded inference in a single inference interface.
The implementation uses established open-source geometric and protein utilities:

| Component | Source | License / location |
|---|---|---|
| Residue encoders, geometric trunk foundations, torsion/atom utilities | [PepFlow](https://github.com/Ced3-han/PepFlowww) | MIT, `licenses/PepFlow-LICENSE` |
| SO(3) and frame-based geometry utilities | [FrameFlow](https://github.com/microsoft/protein-frame-flow) | MIT, `licenses/FrameFlow-LICENSE` |
| Rigid transforms and IPA foundations | [OpenFold](https://github.com/aqlaboratory/openfold) / [AlphaFold](https://github.com/google-deepmind/alphafold) | Apache-2.0, source headers and `licenses/Apache-2.0.txt` |
| Frozen sequence-prior backend (external) | [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | Upstream MIT |
| Cyclic cofolding backend (external) | [ColabDesign](https://github.com/sokrypton/ColabDesign), AfCycDesign cyclic-offset protocol | Upstream MIT; bundled AlphaFold code Apache-2.0 |
| AlphaFold model parameters (external) | Official AlphaFold parameter distribution | Upstream parameter terms |

Relevant methods: PepFlow (ICML 2024), ProteinMPNN (Science 2022), AlphaFold
(Nature 2021), and AfCycDesign. The generator's active upstream-derived code is
organized under `dcflow/model`, `dcflow/geometry`, and `dcflow/vendor/openfold`.
Third-party copyright notices are retained. They identify upstream contributors,
not the anonymous submitting authors.

The 5LSO example is derived from the public Protein Data Bank structure. Example
files contain structural records and an anonymized-header receptor alignment;
they contain no project provenance or submitting-author metadata.
