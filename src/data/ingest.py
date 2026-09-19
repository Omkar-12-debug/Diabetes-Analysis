"""Data ingestion module for diabetes prediction dataset.

Verifies and ingests the raw diabetes prediction dataset into data/raw/.
"""

import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EXPECTED_ROWS = 100000
EXPECTED_COLUMNS = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "smoking_history",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "diabetes",
]


class IngestionError(Exception):
    """Raised when data ingestion or integrity checks fail."""

    pass


def verify_dataset_integrity(df: pd.DataFrame) -> None:
    """Verify that the loaded dataframe satisfies expected row and column criteria.

    Args:
        df: Input pandas DataFrame to verify.

    Raises:
        IngestionError: If row count, column count, or column names mismatch.
    """
    if len(df) != EXPECTED_ROWS:
        raise IngestionError(
            f"Expected {EXPECTED_ROWS} rows, but found {len(df)} rows."
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise IngestionError(
            f"Columns mismatch.\nExpected: {EXPECTED_COLUMNS}\nFound: {list(df.columns)}"
        )


def ingest_data(
    source_path: Optional[str | Path] = None,
    target_path: str | Path = "data/raw/diabetes_prediction_dataset.csv",
) -> pd.DataFrame:
    """Ingest dataset into raw data directory and verify integrity.

    If target_path already exists and source_path is None or identical,
    verifies target_path in place. Otherwise copies from source_path to target_path.

    Args:
        source_path: Optional path to source CSV file.
        target_path: Path where raw dataset should reside.

    Returns:
        pd.DataFrame: The verified raw dataset.

    Raises:
        FileNotFoundError: If neither target nor source exists.
        IngestionError: If file integrity verification fails.
    """
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if source_path is not None:
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source dataset not found at: {source}")
        if source.resolve() != target.resolve():
            logger.info("Copying dataset from %s to %s", source, target)
            shutil.copy2(source, target)

    if not target.exists():
        raise FileNotFoundError(
            f"Target dataset does not exist at: {target}. "
            "Please provide a valid source_path to copy from."
        )

    logger.info("Loading and verifying dataset integrity at %s", target)
    df = pd.read_csv(target)
    verify_dataset_integrity(df)
    logger.info(
        "Dataset successfully verified: %d records, %d columns.",
        len(df),
        len(df.columns),
    )
    return df


if __name__ == "__main__":
    ingest_data()
