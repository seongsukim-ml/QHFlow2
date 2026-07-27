# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Compatibility names for QHFlow2's unified QH9 shard implementation."""

from __future__ import annotations

import argparse
import os
import warnings

warnings.warn(
    "dataset_module.qh9_datasets_shard is deprecated; use dataset_module.qh9_dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dataset_module.qh9_dataset import (
    QH9Dynamic,
    QH9DynamicWriter as QH9Dynamic_shard,
    QH9Stable,
    QH9StableWriter as QH9Stable_shard,
    parse_shard_idx,
)

__all__ = ["QH9Stable", "QH9Dynamic", "QH9Stable_shard", "QH9Dynamic_shard", "parse_shard_idx"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare QH9 LMDB shards")
    parser.add_argument("--root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset"))
    parser.add_argument("--name", choices=("QH9Stable", "QH9Dynamic"), default="QH9Stable")
    parser.add_argument("--version", choices=("100k", "300k"), default="300k")
    parser.add_argument("--split", default="random")
    parser.add_argument("--dynamic_split", choices=("geometry", "mol"), default="geometry")
    parser.add_argument("--shard_num", type=int, default=60)
    parser.add_argument("--shard_idx", default="-1")
    parser.add_argument("--prefix", default="_shard")
    args = parser.parse_args()
    common = dict(root=args.root, shard_num=args.shard_num, shard_idx=parse_shard_idx(args.shard_idx), prefix=args.prefix)
    dataset = QH9Stable(split=args.split, **common) if args.name == "QH9Stable" else QH9Dynamic(split=args.dynamic_split, version=args.version, **common)
    print(dataset)
    print(f"samples={len(dataset)}")


if __name__ == "__main__":
    main()
