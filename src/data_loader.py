import pandas as pd

class DataLoader:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def clean_data(self, data: pd.DataFrame, how: str = 'any', subset: list = None) -> pd.DataFrame:
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
            if not isinstance(data, pd.DataFrame):
                print("Error: Input must be a pandas DataFrame")
                return pd.DataFrame()
            
            # Check if DataFrame is empty
            if data.empty:
                print("Error: DataFrame is empty")
                return pd.DataFrame()
            
            # Clean the data
            cleaned_data = data.dropna(how=how, subset=subset)
            return cleaned_data
        except Exception as e:
            print(f"Error cleaning data: {e}")
            return pd.DataFrame()
    
