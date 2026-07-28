# Third-party notices

QHFlow2 contains or distributes material from the projects and datasets below.
The MIT license for QHFlow2-authored files does not replace these terms.

## AIRS / QHNet

- Project: Efficient and Equivariant Graph Networks for Predicting Quantum
  Hamiltonian (QHNet)
- Source: https://github.com/divelab/AIRS/tree/main/OpenDFT/QHNet
- Upstream license: GPL-3.0-only
- Authors: Haiyang Yu, Zhao Xu, Xiaofeng Qian, Xiaoning Qian, and Shuiwang Ji
- QHFlow2 changes: refactoring, flow-matching integration, new backbones,
  configuration changes, training utilities, and dataset support

The affected paths are identified in `REUSE.toml`.

## QH9 benchmark and dataset

- Project: QH9: A Quantum Hamiltonian Prediction Benchmark for QM9 Molecules
- Source: https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9
- Upstream license: CC-BY-NC-SA-4.0
- Authors: Haiyang Yu, Meng Liu, Youzhi Luo, Alex Strasser, Xiaofeng Qian,
  Xiaoning Qian, and Shuiwang Ji

QHFlow2's MIT-licensed native loader interoperates through the published QH9
data schema and split protocol; the upstream QH9 loader source is not included.
The upstream CC notice in this section applies to the QH9 dataset and other
upstream material actually covered by that notice. It is not used as the
license for QHFlow2 source code. QHFlow2 checkpoint weights trained on QH9 are
separately released under CC-BY-NC-SA-4.0 as described in `ckpts/README.md`.

## FairChem / eSCN

- Project: FairChem
- Source: https://github.com/facebookresearch/fairchem
- Upstream license: MIT
- Copyright: Meta Platforms, Inc. and affiliates
- QHFlow2 changes: Hamiltonian-specific backbones, variants, and integration

The original Meta copyright and MIT notices are retained in the source files.

## SchNOrb Hamiltonian datasets

- Dataset: Hamiltonian datasets for “Unifying machine learning and quantum
  chemistry with a deep neural network for molecular wavefunctions”
- Source: https://depositonce.tu-berlin.de/items/b25de7f8-3c7a-4e41-9ffe-6aca65ada0e5
- Upstream license: MIT
- Authors: Kristof T. Schütt, Michael Gastegger, Alexandre Tkatchenko,
  Klaus-Robert Müller, and Reinhard J. Maurer

The tracked `dataset/water/` and `dataset/water_shard/` trees include the
published `schnorb_hamiltonian_water` archive and derived representations.

## Revised MD17

- Dataset: Revised MD17 (rMD17)
- Source: https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
- Upstream license: CC0-1.0
- Authors: Anders S. Christensen and O. Anatole von Lilienfeld

## nablaDFT

- Project: nablaDFT
- Source: https://github.com/AIRI-Institute/nablaDFT
- Upstream license: MIT

The repository currently contains download and analysis helpers, not the
nablaDFT dataset itself.
