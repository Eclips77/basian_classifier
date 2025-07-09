import pandas as pd
from data_loader import DataLoader

class FileLoader(DataLoader):
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.
        
        Args:
            file_path (str): Path to the CSV file.
        
        Returns:
            pd.DataFrame: Loaded data.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            raise ValueError(f"Error loading data from {file_path}: {e}")
