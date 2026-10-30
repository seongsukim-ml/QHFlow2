# QHFlow2

QHFlow2 predicts density functional theory (DFT) Hamiltonians with equivariant
flow matching. This repository provides model code, pretrained checkpoints,
and tools for evaluating Hamiltonians, energies, forces, and SCF convergence.

**Paper:** [Machine Learning Hamiltonians are Accurate Energy-Force Predictors](https://proceedings.mlr.press/v306/kim26an.html)
— ICML 2026 · [arXiv](https://arxiv.org/abs/2602.16897)

[Quick start](#quick-start) · [Checkpoints](#saved-checkpoints) · [Results](#qh9stable-benchmark-update) · [Reproduction](#reproduction) · [Version history](#version-history) · [Citation](#citation)

## Quick start

Clone the repository and run the commands below from its root:

```bash
git clone https://github.com/seongsukim-ml/QHFlow2.git
cd QHFlow2
```

### Check the sample data

Three QH9Stable test molecules are included (0.377 MB). Checking them needs
only NumPy; no full dataset or model download is required.

```bash
python -m pip install numpy
PYTHONPATH=src python -m reproduction.sample_data check \
  --manifest examples/qh9stable/manifest.json
```

The [sample guide](examples/qh9stable/README.md) explains the molecule IDs,
array formats, and how to run a checkpoint on these samples. These examples
check that loading and inference work; they are not an accuracy benchmark.

### Install the model runtime

For the updated models, use a separate Python 3.12 environment with the
[Stable/Muon dependencies](requirements-stable-muon.txt):

```bash
python -m pip install -r requirements-stable-muon.txt
```

Also install `torch_scatter` and `torch_cluster` extensions matching your
PyTorch/CUDA build; see the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.7.0/install/installation.html).
The [tested runtime notes](docs/QH9STABLE_MUON.md#stable-data-and-runtime)
list the versions used for the updated models. Q-tensor generation uses
PySCF; Psi4 is not required.

Download a checkpoint below, then follow the
[CPU inference example](examples/qh9stable/README.md#run-a-checkpoint-smoke-test)
for a small 1-step or 3-step run. Earlier environment recipes are in the
[legacy installation archive](archive/LEGACY_SETUP.md).

## Saved Checkpoints

**Checkpoints are hosted on [Hugging Face](https://huggingface.co/ksusu/QHFlow2-QH9),
not in Git.** The following updated **v0.2.0-rc.2** models have completed training
and full-test evaluation on QH9Stable (random split).

| Model | Weights |
| --- | --- |
| QHFlow2-s | [Download](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/main/QH9Stable-random/v0.2.0-rc.2/QHFlow2-s/weights.ckpt) |
| QHFlow2-s + O(3) group averaging | [Download](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/main/QH9Stable-random/v0.2.0-rc.2/QHFlow2-O3-s/weights.ckpt) |
| QHFlow2-m | [Download](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/main/QH9Stable-random/v0.2.0-rc.2/QHFlow2-m/weights.ckpt) |
| QHFlow2-l | [Download](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/main/QH9Stable-random/v0.2.0-rc.2/QHFlow2-l/weights.ckpt) |
| QHFlow2-xl | [Download](https://huggingface.co/ksusu/QHFlow2-QH9/resolve/main/QH9Stable-random/v0.2.0-rc.2/QHFlow2-xl/weights.ckpt) |

Each [model folder](https://huggingface.co/ksusu/QHFlow2-QH9/tree/main/QH9Stable-random/v0.2.0-rc.2)
includes inference weights, the exact saved configuration, metrics, provenance,
license, and checksums. Optimizer state is not included. See the
[loading guide](docs/releases/v0.2.0-rc.2-checkpoints.md#loading) for details.

The **16 original QH9 checkpoints** remain in the
[same Hugging Face repository](https://huggingface.co/ksusu/QHFlow2-QH9/tree/main).
The [legacy Google Drive collection](https://drive.google.com/drive/folders/14My7LuvDJiM9hvG2n4_5T9UN97wp4x9w?usp=sharing)
also contains earlier dataset-specific checkpoints. Load original models with
their original configurations; see the [configuration audit](docs/MODEL_CONFIGS.md).

## QH9Stable benchmark update

Updated **2026-10-28** · Updated version **[v0.2.0-rc.2](docs/releases/v0.2.0-rc.2-checkpoints.md)**
(GitHub release is a draft; the checkpoints are available on Hugging Face).

![QH9Stable original-paper and updated-version results](docs/assets/qh9-stable-updated.png)

- **Evaluation:** all 13,084 official random-split test molecules, seed 42,
  with paired **3-step / 1-step** rows for each updated model. H and orbital
  energy errors are in μEh; Sc, SCF ratio, and ERIC are percentages.
- **Model changes:** order fix + **gate change (spectral)** + Muon (1e-3),
  with norm, embedding, and other unsupported parameters on AdamW (1e-4).
  These results combine architecture and training changes.
- **+ O(3) group averaging:** the grey rows use a separately trained model
  with shared-weight identity/inversion averaging.
- **Compute:** inference is measured on one H100 80GB, batch 8, without
  `torch.compile`, at the step count shown. Training GPU hours are reported
  only for the updated models. Inference timing excludes data loading and DFT.
- **SCF:** the original-model ratios use the 300 molecules from original-paper
  Figure 4. ERIC adds one initial Hamiltonian build; it is not a wall-time speedup.
  Dashes indicate unavailable measurements, including updated-model E/F and SCF.

[Full measurement notes and metric definitions](docs/benchmarks/qh9-stable.md) ·
[Unrounded results](reproduce/benchmark/current-results.json) ·
[1-step / 3-step inference measurements](reproduce/README.md#inference-speed)

## Dataset

The bundled samples are enough for a small inference test. Full training and
benchmark evaluation require separately downloaded datasets:

- **QH9Stable / QH9Dynamic:** [download and preprocessing guide](dataset/README.md#qh9).
- **MD17 / rMD17:** [dataset downloads](dataset/README.md#md17-and-rmd17).
- **Updated benchmarks:** [QH9Stable split and runtime](docs/QH9STABLE_MUON.md#stable-data-and-runtime).

Keep full datasets and generated caches outside Git.

## Reproduction

Start with the [reproduction guide](reproduce/README.md). Current release presets
use seed 42, effective batch size 32, and 260,000 optimizer steps.

| Task | Guide |
| --- | --- |
| Train S, O(3)-S, M, L, or XL | [Audited training presets and GPU layouts](reproduce/README.md#training) |
| Load a checkpoint on three sample molecules | [Sample inference](examples/qh9stable/README.md#run-a-checkpoint-smoke-test) |
| Measure 1-step / 3-step inference speed | [Timing protocol and commands](reproduce/README.md#inference-speed) |
| Reproduce SCF and DFT energy/force evaluation | [Evaluation protocol](reproduce/README.md#scf-and-dft-energyforces) |
| Identify the molecules used in Figure 4 | [300-molecule list](reproduce/scf/recorded-300.csv) and [provenance](reproduce/scf/provenance.json) |
| Inspect model sizes and checkpoint compatibility | [Configuration audit](docs/MODEL_CONFIGS.md) |

Current YAML presets are in `src/config_qh9/`; evaluation protocols and recorded
results are in `reproduce/`. Older training configs, notebooks, and exploratory
scripts are kept in [archive/](archive/README.md), outside the current config search path.

## Version History

| Version | Date | Status and changes |
| --- | --- | --- |
| [v0.2.0-rc.2](docs/releases/v0.2.0-rc.2-checkpoints.md) | Updated 2026-10-28 | Draft release; five completed checkpoints on Hugging Face, updated benchmarks, and [compute optimizations](docs/releases/v0.2.0-rc.2.md). |
| [v0.2.0-rc.1](https://github.com/seongsukim-ml/QHFlow2/releases/tag/v0.2.0-rc.1) | 2026-10-14 | Published code prerelease: PySCF, corrected SO(3)/O(3) models, and Stable/Muon training. |
| [ver.0.1.0](https://github.com/seongsukim-ml/QHFlow2/releases/tag/ver.0.1.0) | 2026-10-07 | Original model implementations, preserved for reproducibility. |

For v0.2.0, S/M/L/XL and O(3)-S benchmarks are complete. **O(3)-M/L/XL**
Muon training and benchmarks remain planned.

<details>
<summary>Equivariance correction and checkpoint compatibility</summary>

A feature-ordering bug arose in QHFlow2 while adapting conventions from prior
models. It broke rotational equivariance for multi-channel spherical features.
Follow-up experiments showed similar performance after correcting the ordering;
the updated benchmarks also include spectral gating and optimizer changes and
are not an ablation of that correction alone.

Legacy model definitions are retained to load original checkpoints. The complete
corrected models (`QHFlow2_order_fixed` and `QHFlow2_O3`) require newly trained
weights. S/M/L layer counts are unchanged; the current XL preset matches the
original five-layer checkpoint. Older four-layer XL weights need their saved
configuration.

See [equivariance history and validation](docs/EQUIVARIANCE_STATUS.md),
[release notes](docs/releases/v0.2.0-rc.1.md), and the
[checkpoint/configuration audit](docs/MODEL_CONFIGS.md).

</details>

[Full changelog](CHANGELOG.md) · [Release guide](docs/RELEASING.md) ·
[Git history cleanup](docs/history/2026-10-28-cleanup.md)

## Citation

If you use QHFlow2, please cite the [ICML 2026 paper](https://proceedings.mlr.press/v306/kim26an.html):

```bibtex
@inproceedings{pmlr-v306-kim26an,
  title = {Machine Learning {H}amiltonians are Accurate Energy-Force Predictors},
  author = {Kim, Seongsu and Lee, Chanhui and Kim, Yoonho and Yun, Seongjun and Kim, Honghui and Kim, Nayoung and Park, Changyoung and Han, Sehui and Lim, Sungbin and Ahn, Sungsoo},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  pages = {58218--58253},
  year = {2026},
  volume = {306},
  series = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v306/kim26an.html}
}
```

For the original QHFlow method:

```bibtex
@inproceedings{kim2025high,
  title={High-order Equivariant Flow Matching for Density Functional Theory Hamiltonian Prediction},
  author={Kim, Seongsu and Kim, Nayoung and Kim, Dongwoo and Ahn, Sungsoo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Acknowledgements and license

This project builds on [AIRS / QHNet](https://github.com/divelab/AIRS),
[QH9](https://github.com/divelab/AIRS/tree/main/OpenDFT/QHBench/QH9), and
[rMD17](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038).

Licensing is per file: original QHFlow2 source is MIT; identified AIRS/QHNet-derived
software remains GPL-3.0-only. QH9-trained checkpoint weights and the bundled
QH9 sample data use CC-BY-NC-SA-4.0. See [LICENSE](LICENSE), [REUSE.toml](REUSE.toml),
and the [licensing guide](docs/legal/README.md) for attribution and dataset terms.
