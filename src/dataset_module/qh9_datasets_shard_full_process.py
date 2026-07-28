# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Deprecated import path for the unified QH9 shard implementation."""

import warnings

warnings.warn(
    "dataset_module.qh9_datasets_shard_full_process is deprecated; use dataset_module.qh9_dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dataset_module.qh9_datasets_shard import (
    QH9Dynamic,
    QH9Dynamic_shard,
    QH9Stable,
    QH9Stable_shard,
    main,
    parse_shard_idx,
)

__all__ = ["QH9Stable", "QH9Dynamic", "QH9Stable_shard", "QH9Dynamic_shard", "parse_shard_idx"]

if __name__ == "__main__":
    main()
