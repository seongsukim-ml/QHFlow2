# SPDX-FileCopyrightText: 2026 Seongsu Kim
# SPDX-License-Identifier: MIT
"""Explicit, read-only access to existing QH9Stable shards and split files."""

import json
from pathlib import Path

import torch

from .qh9_dataset import QH9Stable


class QH9StableReadonly(QH9Stable):
    def __init__(self, processed_directory, split_file, **kwargs):
        self._processed_directory = str(Path(processed_directory).resolve(strict=True))
        self._split_file = str(Path(split_file).resolve(strict=True))
        for name in ("index.json", "ALL_SHARDS_COMPLETED.txt"):
            if not (Path(self._processed_directory) / name).is_file():
                raise FileNotFoundError(f"Missing completed QH9Stable metadata: {name}")
        super().__init__(root=str(Path(self._processed_directory).parent.parent), **kwargs)

    @property
    def processed_dir(self):
        return self._processed_directory

    def _download(self):
        pass  # This reader only opens previously processed, validated files.

    def _process(self):
        pass

    def download(self):
        raise RuntimeError("QH9StableReadonly cannot download data")

    def process(self):
        raise RuntimeError("QH9StableReadonly cannot modify data")

    def _load_metadata(self):
        with open(self._split_file) as handle:
            split = json.load(handle)
        with open(Path(self.processed_dir) / "index.json") as handle:
            rows = json.load(handle)["index"]
        self._index = [(int(a), int(b), int(c)) for a, b, c in rows]
        joined = [int(i) for key in ("train", "val", "test") for i in split[key]]
        if len(joined) != len(rows) or sorted(joined) != list(range(len(rows))):
            raise ValueError("QH9Stable split must cover every row exactly once")
        for shard in {row[0] for row in self._index}:
            if not (Path(self.processed_dir) / "lmdbs" / f"shard_{shard:03d}.lmdb" / "data.mdb").is_file():
                raise FileNotFoundError(f"Missing QH9Stable shard {shard}")
        self.train_mask = torch.tensor(split["train"], dtype=torch.long)
        self.val_mask = torch.tensor(split["val"], dtype=torch.long)
        self.test_mask = torch.tensor(split["test"], dtype=torch.long)
        self.slices = {"id": torch.arange(len(self._index) + 1)}
