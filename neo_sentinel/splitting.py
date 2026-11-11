"""Train / validation / test split that keeps each asteroid in one part only.

The dataset has about 3.3 rows per asteroid (one per recorded close
approach) and the label belongs to the asteroid. A plain random row split
puts the same asteroid in train and test, so the model is partly graded on
objects it has already seen. Splitting by asteroid id removes that leak.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from neo_sentinel.config import RANDOM_STATE

N_SLICES = 20  # 20 slices of 5% each: 14 train, 3 validation, 3 test


def group_split(y, groups, val_slices: int = 3, test_slices: int = 3, seed: int = RANDOM_STATE):
    """Return (train_idx, val_idx, test_idx), stratified on y, grouped by asteroid."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    n_slices = min(N_SLICES, len(np.unique(groups)))
    if val_slices + test_slices >= n_slices:
        raise ValueError("Not enough asteroids to make a train, validation and test split.")

    splitter = StratifiedGroupKFold(n_splits=n_slices, shuffle=True, random_state=seed)
    slices = [held_out for _, held_out in splitter.split(np.zeros(len(y)), y, groups)]

    test_idx = np.concatenate(slices[:test_slices])
    val_idx = np.concatenate(slices[test_slices : test_slices + val_slices])
    train_idx = np.concatenate(slices[test_slices + val_slices :])
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def group_cv(n_splits: int = 3, seed: int = RANDOM_STATE) -> StratifiedGroupKFold:
    """Cross validation for tuning, also grouped by asteroid."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
