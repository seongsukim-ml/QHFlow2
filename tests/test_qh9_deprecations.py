# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.mark.parametrize(
    ("module_name", "attribute"),
    [
        ("dataset_module.qh9_common", "BaseQH9Dataset"),
        ("dataset_module.qh9_datasets_split", "QH9Stable"),
        ("dataset_module.qh9_datasets_shard", "QH9Stable_shard"),
        ("dataset_module.qh9_datasets_shard_full_process", "QH9Stable_shard"),
    ],
)
def test_legacy_qh9_loader_modules_warn_and_keep_exports(module_name, attribute):
    sys.modules.pop(module_name, None)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        module = importlib.import_module(module_name)

    assert hasattr(module, attribute)
