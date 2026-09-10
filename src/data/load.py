from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config import RAW_DATA_DIR


DATA_FILE = RAW_DATA_DIR / "twcs.csv"


def _check_dataset_exists() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_FILE}"
        )


def load_sample(nrows: int = 10) -> pd.DataFrame:
    """
    Load a small sample of the raw Twitter support dataset.
    """
    _check_dataset_exists()

    return pd.read_csv(DATA_FILE, nrows=nrows)


def load_data(
    nrows: int | None = None,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the Twitter support dataset.

    Parameters
    ----------
    nrows:
        Number of rows to load. If None, loads the complete dataset.
    usecols:
        Optional list of columns to load.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    _check_dataset_exists()

    return pd.read_csv(
        DATA_FILE,
        nrows=nrows,
        usecols=usecols,
    )


def iter_data(
    chunksize: int = 100_000,
    usecols: list[str] | None = None,
) -> Iterator[pd.DataFrame]:
    """
    Iterate over the dataset in chunks.

    The complete TWCS dataset is large, so chunked processing
    avoids loading the entire CSV into memory at once.
    """
    _check_dataset_exists()

    return pd.read_csv(
        DATA_FILE,
        usecols=usecols,
        chunksize=chunksize,
    )