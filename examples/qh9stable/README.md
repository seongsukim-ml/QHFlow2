# Three QH9Stable test samples

These are the first three entries of the recorded official QH9Stable random
**test** split, exported for loading and inference smoke tests. They are not a
training split or a representative accuracy benchmark. All three compressed
NPZ files together occupy **376,906 bytes (0.377 MB)**. No checkpoints or full
SQLite/LMDB databases are included.

| Dataset index (zero-based) | Source record ID | Test position | Formula | Atoms | AOs |
| ---: | ---: | ---: | --- | ---: | ---: |
| 11 | 11 | 0 | CH3NO | 6 | 57 |
| 24 | 24 | 1 | C2N2 | 4 | 56 |
| 31 | 31 | 2 | C2H4N2 | 8 | 76 |

`manifest.json` records stable molecule identities, source record and split
hashes, units, conventions and per-file checksums. Dataset indices are not
asserted to be QM9 molecule names. Coordinates are in Angstrom; Hamiltonians
are in Hartree; overlap is dimensionless; the basis is def2-SVP.

## Check or read without the full dataset

From the repository root, the check command needs only NumPy:

```bash
PYTHONPATH=src python -m reproduction.sample_data check \
  --manifest examples/qh9stable/manifest.json
```

Raw/reference arrays use native PySCF AO order:

```python
import numpy as np

with np.load("examples/qh9stable/sample_000011.npz", allow_pickle=False) as sample:
    atoms = sample["atoms"]                 # atomic numbers, (N,)
    positions = sample["pos_ang"]           # Angstrom, (N, 3)
    reference_h = sample["H_reference"]     # reference Hamiltonian, (AO, AO)
    overlap = sample["S"]                   # native PySCF overlap
    initial_h = sample["H_initial"]         # stored initialization Hamiltonian
```

Keys prefixed `input__` are the exact numerical PyG fields returned by the
existing read-only QH9 loader, including padded 14-orbital Hamiltonian, overlap,
initial-H blocks, masks and edge indices. They use the loader's e3nn AO
transformation. Q tensors and orbital-coefficient caches are omitted, matching
the published checkpoint inference path. Floating arrays retain float64;
integer arrays retain int64. `as_pyg` reconstructs the graph without a database:

```python
from reproduction.sample_data import read_samples, as_pyg

manifest, samples = read_samples("examples/qh9stable/manifest.json")
graphs = [as_pyg(arrays) for record, arrays in samples]
```

## Run a checkpoint smoke test

Install the [Stable/Muon runtime](../../docs/QH9STABLE_MUON.md#stable-data-and-runtime).
Download the updated [QHFlow2-s inference checkpoint](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/e624b5dee4cb2cb29985a34d55bd97359ff0aa55/QH9Stable-random/v0.2.0-rc.2/QHFlow2-s/weights.ckpt)
separately; weights remain on Hugging Face.

```bash
PYTHONPATH=src OMP_NUM_THREADS=2 python -m reproduction.sample_data infer \
  --manifest examples/qh9stable/manifest.json \
  --checkpoint /absolute/path/to/weights.ckpt \
  --device cpu --ode-steps 1 \
  --output /absolute/path/to/a-new-output-directory
```

Use `--ode-steps 3` for three-step inference. This loads the exported weights
with `weights_only=True` and `strict=True`, runs all three molecules and writes
native-AO predictions, reference matrices and per-molecule H MAE in a fresh
output directory. On the shared SC26 workspace, put outputs below
`/dataset/seongsu/shared-home/workspace/project/MALOQ/outputs/`.
The small CPU run checks functionality; its accuracy and timing are not the
published full-test benchmark. It does not start training or download data.

## Regenerate from an existing QH9 dataset

```bash
PYTHONPATH=src python -m reproduction.sample_data export \
  --processed-directory /absolute/path/to/QH9Stable/processed \
  --split-file /absolute/path/to/processed_QH9Stable_random_12.json \
  --output /absolute/path/to/a-new-sample-directory
```

Use the split hash in the manifest to reproduce this exact selection. The
exporter reads trusted local shards, preserves all numerical arrays and does
not alter the source dataset. NPZ archive bytes may differ across NumPy/ZIP
versions; compare array values as well as the recorded source hashes.

## Attribution and license

Subset of [QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9)
by Haiyang Yu and the QH9 contributors; see [CITATION.bib](CITATION.bib).
QHFlow2 export and model-input representations by Seongsu Kim, 2026.
Changes: selection of three test rows, lossless NPZ packaging and the existing
QHFlow2 AO transformation/block padding. No new DFT labels were computed.

The sample data and derived numerical representations retain
**CC-BY-NC-SA-4.0**; see [LICENSE.txt](LICENSE.txt). They are not MIT-licensed
source code. Scripts remain under their source-code license.
