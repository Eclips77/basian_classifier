import pandas as pd

class DataLoader:

    def __init__(self, file_path: str, label_column: str):
        self.file_path = file_path
        self.__label_column = label_column
        self.__data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from a CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the loaded data, or an empty DataFrame if an error occurs.
        """
        try:
            # Load the data from the CSV file
            self.__data = pd.read_csv(self.file_path)
            return self.__data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def clean_data(self,  how: str = 'any', subset: list = None) -> pd.DataFrame:
        """
        Clean the DataFrame by removing rows with missing values.
        
        Args:
            data (pd.DataFrame): The DataFrame to clean.
            how (str): 'any' or 'all' to determine how to handle missing values. Default is 'any'.
            subset (list): List of columns to check for missing values. Default is None (all columns).
        
        Returns:
            pd.DataFrame: Cleaned DataFrame with no missing values, or an empty DataFrame if an error occurs.
        """
        try:
            # Check if input is a valid DataFrame
            if not isinstance(self.__data, pd.DataFrame):
                print("Error: Input must be a pandas DataFrame")
                return pd.DataFrame()
            
            # Check if DataFrame is empty
            if self.__data.empty:
                print("Error: DataFrame is empty")
                return pd.DataFrame()
            
            # Clean the data
            self.__data = self.__data.dropna(how=how, subset=subset)
            return self.__data
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return pd.DataFrame()
    
    def get_data(self) -> pd.DataFrame:
        return self.__data

    def get_features(self) -> pd.DataFrame:
        """
        Get the features from the DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing the features, or an empty DataFrame if no data is loaded.
        """
        if self.__data is not None:
            if self.__label_column in self.__data.columns:
                # Drop the label column to return only features
                return self.__data.drop(columns=[self.__label_column], errors='ignore')
        return pd.DataFrame()

    def get_labels(self) -> pd.Series:
        """
        Get the labels from the DataFrame.

        Returns:
            pd.Series: Series containing the labels, or an empty Series if no data is loaded.
        """
        if self.__data is not None and self.__label_column in self.__data.columns:
            return self.__data[self.__label_column]
        return pd.Series(dtype=float)
