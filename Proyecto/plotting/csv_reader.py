from pathlib import Path

import numpy as np
import polars as pl


def load_lengths_from_csv_dir(directory, column_length: int = 1500, tail: bool = True):
    """Load the 'l' (length) column from all CSV files in a directory using polars and pathlib.

    Args:
        directory (str): The directory containing the CSV files.
        column_length (int): The number of entries to take from the end of each file.
        tail (bool): Whether to take the last entries or the first.

    Returns:
        np.ndarray: A numpy array of shape (n_runs, n_episodes).

    """
    directory = Path(directory)
    lengths = []
    for csv_file in sorted(directory.glob('*.csv')):
        df = pl.read_csv(csv_file, skip_lines=1)
        if 'l' in df.columns:
            # Only append if length is sufficient
            if len(df) >= column_length:
                if tail:
                    lengths.append(df['l'].tail(column_length).to_numpy())
                else:
                    lengths.append(df['l'].head(column_length).to_numpy())

    return np.array(lengths)
