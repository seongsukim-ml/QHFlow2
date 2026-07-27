# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

import pickle
import sqlite3
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_module.qh9_dataset import (
    QH9SQLiteShardWriter,
    _layout,
    _stable_split,
)


def test_stable_split_does_not_fetch_hamiltonian_blobs():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE data (id INTEGER, N INTEGER, Ham BLOB)")
    connection.executemany(
        "INSERT INTO data VALUES (?, ?, ?)",
        ((index, 5, b"large Hamiltonian payload") for index in range(10)),
    )
    queries = []
    connection.set_trace_callback(queries.append)
    try:
        _stable_split(_layout("QH9Stable", "", "size_ood"), connection)
    finally:
        connection.close()

    assert not any("SELECT *" in query.upper() for query in queries)


def test_writer_copies_sqlite_buffers_before_torch_conversion(monkeypatch, tmp_path):
    class FakeMeanField:
        @staticmethod
        def energy_tot(_density):
            return -1.25

    overlap = np.eye(5, dtype=np.float64)
    initial = np.eye(5, dtype=np.float64) * 2
    monkeypatch.setattr(
        "dataset_module.qh9_dataset.calc_overlap_and_init_hamiltonian",
        lambda *_args, **_kwargs: (overlap, initial, FakeMeanField()),
    )
    monkeypatch.setattr(
        "dataset_module.qh9_dataset.cal_orbital_and_energies",
        lambda *_args, **_kwargs: (torch.zeros((1, 5)), torch.eye(5).unsqueeze(0)),
    )
    monkeypatch.setattr(
        "dataset_module.qh9_dataset.calc_dm0",
        lambda *_args, **_kwargs: np.eye(5, dtype=np.float64),
    )
    writer = QH9SQLiteShardWriter(
        layout=_layout("QH9Stable", "", "random"),
        root_path=str(tmp_path / "raw.db"),
        shard_num=1,
        save_path=str(tmp_path),
        include_dft_energy=True,
    )
    row = (
        0,
        1,
        np.asarray([1], dtype=np.int32).tobytes(),
        np.zeros((1, 3), dtype=np.float64).tobytes(),
        np.eye(5, dtype=np.float64).tobytes(),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="The given NumPy array is not writable")
        _, payload = writer._encode((0, row))

    assert pickle.loads(payload)["dft_energy"] == pytest.approx(-1.25)
