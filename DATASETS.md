# Dataset licenses and distribution

Datasets are not covered by the default MIT license for QHFlow2 source code.
Each dataset retains its own upstream terms.

| Dataset distributed or referenced by QHFlow2 | Source | License | Redistribution status |
| --- | --- | --- | --- |
| SchNOrb Hamiltonian MD17-style data (`water`, `ethanol`, `malondialdehyde`, `uracil`) | [TU Berlin DepositOnce](https://depositonce.tu-berlin.de/items/b25de7f8-3c7a-4e41-9ffe-6aca65ada0e5) | MIT | Allowed with the upstream notice and citation |
| Revised MD17 (`rmd-*`) | [Figshare rMD17 v3](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038) | CC0-1.0 | Allowed; citation is strongly requested |
| QH9 Stable and Dynamic | [AIRS/QHBench/QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9) | CC-BY-NC-SA-4.0 | Non-commercial only; attribution, change indication, and ShareAlike apply |
| nablaDFT | [AIRI-Institute/nablaDFT](https://github.com/AIRI-Institute/nablaDFT) | MIT | Follow the upstream notice and dataset citation |

## Data currently tracked in Git

This repository tracks the SchNOrb water raw archive/database and processed
PyTorch/LMDB forms under:

- `dataset/water/`
- `dataset/water_shard/`

The original SchNOrb MIT terms continue to apply to the raw data. QHFlow2's
generated indexes, splits, and container metadata are also released under MIT,
without changing the license of the underlying records.

## Google Drive packages

The linked Drive folders contain processed copies but currently do not include
per-folder license sidecars. Before redistributing a folder, include:

1. a copy of the applicable license from `LICENSES/`;
2. this dataset card or an equivalent README;
3. the upstream source URL and citation;
4. a description of QHFlow2 processing and any changes; and
5. checksums for the released files.

Do not copy QH9 data into an MIT-only package. QH9 remains non-commercial under
CC-BY-NC-SA-4.0.

This is the QH9 dataset's upstream license, not a CC license grant for QHFlow2
source code. Among artifacts authored and released by QHFlow2, CC-BY-NC-SA-4.0
is assigned only to checkpoint weights trained on QH9.
