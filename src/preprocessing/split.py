"""Stratified train/validation/test splitting with deduplication support.

Splits the diabetes dataset into 70/15/15 partitions while preserving
the ~8.5% positive class distribution via stratified sampling.
"""

import logging
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "diabetes",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
    drop_duplicates: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/validation/test splitting.

    Optionally drops exact duplicate rows before splitting to prevent
    identical records from leaking across partitions.

    Args:
        df: Input DataFrame containing features and target.
        target_col: Name of the binary target column.
        train_ratio: Proportion for training set (default 0.70).
        val_ratio: Proportion for validation set (default 0.15).
        test_ratio: Proportion for test set (default 0.15).
        random_state: Random seed for reproducibility.
        drop_duplicates: If True, remove exact duplicate rows before splitting.

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames.

    Raises:
        ValueError: If ratios do not sum to 1.0.
    """
    total = round(train_ratio + val_ratio + test_ratio, 10)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    working_df = df.copy()

    if drop_duplicates:
        before = len(working_df)
        working_df = working_df.drop_duplicates().reset_index(drop=True)
        removed = before - len(working_df)
        logger.info(
            "Deduplication: removed %d exact duplicates (%d -> %d rows).",
            removed,
            before,
            len(working_df),
        )

    y = working_df[target_col]

    # Stage 1: Split into train and temp (val + test)
    temp_ratio = val_ratio + test_ratio  # 0.30
    train_df, temp_df = train_test_split(
        working_df,
        test_size=temp_ratio,
        stratify=y,
        random_state=random_state,
    )

    # Stage 2: Split temp into validation and test (50/50 of 0.30 = 0.15 each)
    relative_test_ratio = test_ratio / temp_ratio
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        stratify=temp_df[target_col],
        random_state=random_state,
    )

    # Reset indices for clean downstream usage
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Stratified split complete: train=%d, val=%d, test=%d (total=%d).",
        len(train_df),
        len(val_df),
        len(test_df),
        len(train_df) + len(val_df) + len(test_df),
    )

    # Log class distributions per split
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pos_rate = split_df[target_col].mean() * 100
        logger.info("  %s: %.2f%% positive class (%d / %d).", name, pos_rate,
                     int(split_df[target_col].sum()), len(split_df))

    return train_df, val_df, test_df
