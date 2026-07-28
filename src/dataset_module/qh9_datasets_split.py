# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Backward-compatible entry point for the unified QHFlow2 QH9 loader."""

from __future__ import annotations

import argparse
import os
import warnings

warnings.warn(
    "dataset_module.qh9_datasets_split is deprecated; use dataset_module.qh9_dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from common.dft_utils import calc_overlap_and_init_hamiltonian
from common.matrix_transforms import matrix_transform_single
from common.metric import cal_orbital_and_energies
from dataset_module.qh9_dataset import QH9Dynamic as _QH9Dynamic
from dataset_module.qh9_dataset import QH9Stable as _QH9Stable


class QH9Stable(_QH9Stable):
    def __init__(self, *args, num_chunks=60, chunk_idx=-1, cal_orbital_and_energies=False, **kwargs):
        kwargs.setdefault("return_orbital_and_energies", cal_orbital_and_energies)
        kwargs.setdefault("include_dft_energy", False)
        kwargs.setdefault("include_dft_forces", False)
        kwargs.setdefault("include_cut_orbital_coefficients", False)
        kwargs.setdefault("shard_num", num_chunks)
        kwargs.setdefault("shard_idx", chunk_idx)
        super().__init__(*args, **kwargs)


class QH9Dynamic(_QH9Dynamic):
    def __init__(self, *args, task="", num_chunks=60, chunk_idx=-1, cal_orbital_and_energies=False, **kwargs):
        if task:
            raise ValueError("QH9Dynamic task must be empty")
        kwargs.setdefault("return_orbital_and_energies", cal_orbital_and_energies)
        kwargs.setdefault("include_dft_energy", False)
        kwargs.setdefault("include_dft_forces", False)
        kwargs.setdefault("include_cut_orbital_coefficients", False)
        kwargs.setdefault("shard_num", num_chunks)
        kwargs.setdefault("shard_idx", chunk_idx)
        super().__init__(*args, **kwargs)


def matrix_transform(matrices, atoms, convention="pyscf_def2svp"):
    return matrix_transform_single(matrices, atoms, convention=convention)


def calc_ovlp_and_ham_init(atoms, pos, init="minao"):
    return calc_overlap_and_init_hamiltonian(atoms, pos, init=init, basis="def2svp", xc="b3lyp")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QH9 LMDB shards")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"))
    parser.add_argument("--name", choices=("QH9Stable", "QH9Dynamic"), default="QH9Stable")
    parser.add_argument("--version", choices=("100k", "300k"), default="300k")
    parser.add_argument("--split", default="random")
    parser.add_argument("--num_chunks", type=int, default=60)
    parser.add_argument("--chunk_idx", type=int, default=-1)
    args = parser.parse_args()
    if args.name == "QH9Stable":
        dataset = QH9Stable(root=args.root, split=args.split, num_chunks=args.num_chunks, chunk_idx=args.chunk_idx)
    else:
        dataset = QH9Dynamic(root=args.root, split=args.split, version=args.version, num_chunks=args.num_chunks, chunk_idx=args.chunk_idx)
    print(dataset)
    print(f"samples={len(dataset)}")


if __name__ == "__main__":
    main()
