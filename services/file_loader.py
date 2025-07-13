"""Concrete implementation of ``DataLoader`` for local CSV files."""

import os
import pandas as pd
from services.data_loader import DataLoader

class FileLoader(DataLoader):
    """Load CSV files from disk."""

    DEFAULT_DATA_DIR = "Data"

    @staticmethod
    def select_data_file(data_dir: str = None) -> str:
        """Interactively select a data file.

        Parameters
        ----------
        data_dir : str, optional
            Directory to search for CSV files. Defaults to ``FileLoader.DEFAULT_DATA_DIR``.

        Returns
        -------
        str
            The chosen file path or ``None`` if selection failed.
        """

        data_dir = data_dir or FileLoader.DEFAULT_DATA_DIR
        if not os.path.isdir(data_dir):
            print(f"Data directory not found: {data_dir}")
            return None

        csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
        if not csv_files:
            print(f"No CSV files found in {data_dir}")
            return None

        print("Available data files:")
        for idx, fname in enumerate(csv_files, start=1):
            print(f"{idx}. {fname}")
        print("0. Enter a custom path")

        choice = input("Select file number: ").strip()
        if choice == "0":
            return input("Enter full CSV path: ").strip()

        try:
            idx = int(choice)
            if 1 <= idx <= len(csv_files):
                return os.path.join(data_dir, csv_files[idx - 1])
        except ValueError:
            pass

        print("Invalid selection.")
        return None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load a CSV file into a ``DataFrame``.

        Raises a ``ValueError`` if loading fails so the caller can handle the
        error appropriately.
        """

        if not file_path:
            raise ValueError("No file path provided")

        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise ValueError(f"File not found: {file_path}")
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(f"Error loading data from {file_path}: {exc}") from exc

        if data.empty:
            raise ValueError(f"Loaded DataFrame from {file_path} is empty")

        return data
