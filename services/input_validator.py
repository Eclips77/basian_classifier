# Validator.py
import os
import pandas as pd
from typing import Union

class DataValidator:
    @staticmethod
    def validate_cli(filepath: str, label_identifier: Union[str, int]) -> tuple:
        if not os.path.isfile(filepath):
            print(f"File not found: {filepath}")

        df = pd.read_csv(filepath, nrows=0)  

        columns = list(df.columns)
        if isinstance(label_identifier, int):
            if label_identifier < 0 or label_identifier >= len(columns):
                print(f"Label index {label_identifier} out of range.")
            label_name = columns[label_identifier]
        elif isinstance(label_identifier, str):
            if label_identifier not in columns:
                print(f"Label column '{label_identifier}' not found.")
            label_name = label_identifier
        else:
            print("Label identifier must be string or integer.")

        return filepath, label_name
