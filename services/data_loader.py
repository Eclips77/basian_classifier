import os
import pandas as pd

class DataLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.__data = None



    def load_data(self) -> pd.DataFrame:
        """Load data from a CSV file into a pandas DataFrame.

        Returns the loaded :class:`pandas.DataFrame`.

        Raises:
            FileNotFoundError: If the file does not exist.
            pandas.errors.ParserError: If the CSV cannot be parsed.
        """

        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Any parsing errors will be raised to the caller
        self.__data = pd.read_csv(self.file_path)
        return self.__data


    
    def get_data(self) -> pd.DataFrame:
        return self.__data

