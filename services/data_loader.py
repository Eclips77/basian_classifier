import pandas as pd

class DataLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.__data = None



    def load_data(self):
        """
        Load data from a CSV file into a pandas DataFrame.
        """
        try:
            # Load the data from the CSV file
            self.__data = pd.read_csv(self.file_path)
        except Exception as e:
            print(f"Error loading data: {e}")

    
    def get_data(self) -> pd.DataFrame:
        return self.__data

