# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset_module.qh9_dataset import (
    QH9Layout,
    _dynamic_split,
    _layout,
    _stable_split,
    parse_shard_idx,
)


def _stable_connection(sizes):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE data (id INTEGER, num_nodes INTEGER)")
    connection.executemany("INSERT INTO data VALUES (?, ?)", enumerate(sizes))
    return connection


def test_parse_shard_indices():
    assert parse_shard_idx("-1") == -1
    assert parse_shard_idx("3") == [3]
    assert parse_shard_idx("1, 4,7") == [1, 4, 7]
    assert parse_shard_idx("2-5") == [2, 3, 4, 5]
    with pytest.raises(ValueError, match="increasing"):
        parse_shard_idx("5-2")


def test_stable_splits_are_deterministic_and_size_ood_is_disjoint():
    connection = _stable_connection([20, 21, 22, 23, 19, 24, 18, 21, 22, 23])
    try:
        size_split = _stable_split(_layout("QH9Stable", "", "size_ood"), connection)
        first_random = _stable_split(_layout("QH9Stable", "", "random"), connection)
        second_random = _stable_split(_layout("QH9Stable", "", "random"), connection)
    finally:
        connection.close()

    assert size_split["train"] == [0, 4, 6]
    assert size_split["val"] == [1, 2, 7, 8]
    assert size_split["test"] == [3, 5, 9]
    assert first_random == second_random
    expected_random = np.random.RandomState(43).permutation(10).tolist()
    assert first_random["train"] == expected_random[:8]
    assert first_random["val"] == expected_random[8:9]
    assert first_random["test"] == expected_random[9:]
    assert [len(first_random[key]) for key in ("train", "val", "test")] == [8, 1, 1]
    assert set().union(*map(set, first_random.values())) == set(range(10))


def test_dynamic_splits_preserve_geometry_and_molecule_protocols():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE data (id INTEGER)")
    connection.executemany("INSERT INTO data VALUES (?)", ((index,) for index in range(1_000)))
    try:
        geometry = _dynamic_split(_layout("QH9Dynamic", "300k", "geometry"), connection)
        molecule = _dynamic_split(_layout("QH9Dynamic", "300k", "mol"), connection)
    finally:
        connection.close()

    first_geometry_order = np.random.RandomState(99).permutation(100).tolist()
    assert geometry["train"][:80] == first_geometry_order[:80]
    assert geometry["val"][:10] == first_geometry_order[80:90]
    assert geometry["test"][:10] == first_geometry_order[90:]
    assert molecule["train"] == sorted(molecule["train"])
    assert molecule["val"] == sorted(molecule["val"])
    assert [len(geometry[key]) for key in ("train", "val", "test")] == [800, 100, 100]
    assert [len(molecule[key]) for key in ("train", "val", "test")] == [800, 100, 100]
    assert set().union(*map(set, geometry.values())) == set(range(1_000))
    assert set().union(*map(set, molecule.values())) == set(range(1_000))


def test_layout_rejects_unsupported_combinations():
    with pytest.raises(ValueError, match="QH9Stable split"):
        _layout("QH9Stable", "", "geometry")
    with pytest.raises(ValueError, match="QH9Dynamic version"):
        _layout("QH9Dynamic", "invalid", "geometry")


def test_dynamic_layout_decodes_published_column_positions():
    layout = QH9Layout("QH9Dynamic", "300k", "geometry")
    record_id = np.asarray([7], dtype=np.int64).tobytes()
    row = (record_id, b"unused", 2, b"atoms", b"positions", None, None, None, None, b"hamiltonian")
    decoded = layout.decode_row(row)
    assert decoded == (7, 2, b"atoms", b"positions", b"hamiltonian")
