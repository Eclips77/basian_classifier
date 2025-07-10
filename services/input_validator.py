# Validator.py
import os
import pandas as pd
from typing import Union

class DataValidator:
    """Validate command line input for file path and label column."""

    @staticmethod
    def validate_cli(filepath: str, label_identifier: Union[str, int]) -> tuple:
        """Validate the provided file path and label column.

        Args:
            filepath: Path to the CSV file.
            label_identifier: Column name or index of the label column.

        Returns:
            Tuple containing the validated file path and label column name.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the label column is invalid.
        """

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pd.read_csv(filepath, nrows=0)
        columns = list(df.columns)

        if isinstance(label_identifier, int):
            if label_identifier < 0 or label_identifier >= len(columns):
                raise ValueError(f"Label index {label_identifier} out of range.")
            label_name = columns[label_identifier]
        elif isinstance(label_identifier, str):
            if label_identifier not in columns:
                raise ValueError(f"Label column '{label_identifier}' not found.")
            label_name = label_identifier
        else:
            raise ValueError("Label identifier must be string or integer.")

        return filepath, label_name
