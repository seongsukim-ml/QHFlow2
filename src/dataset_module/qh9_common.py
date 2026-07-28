# SPDX-FileCopyrightText: 2026 Seongsu Kim
#
# SPDX-License-Identifier: MIT

"""Compatibility exports for QHFlow2's native QH9 implementation."""

import warnings

warnings.warn(
    "dataset_module.qh9_common is deprecated; import from dataset_module.qh9_dataset instead.",
    DeprecationWarning,
    stacklevel=2,
)

from dataset_module.qh9_dataset import BaseQH9Dataset, TensorListWrapper

__all__ = ["BaseQH9Dataset", "TensorListWrapper"]
