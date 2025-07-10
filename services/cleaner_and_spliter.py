import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Iterable

class Cleaner:

    def __init__(self, data: pd.DataFrame, label_column: str):
        """Initialize the cleaner with a DataFrame and label column."""

        self.__data = data
        self.__label_column = label_column


        
    def clean_data(self, how: str = "any", subset: Iterable[str] | None = None) -> pd.DataFrame:
        """Remove rows with missing values from the loaded data."""

        if not isinstance(self.__data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if self.__data.empty:
            raise ValueError("DataFrame is empty")

        self.__data = self.__data.dropna(how=how, subset=subset)
        return self.__data
        
    def get_features(self) -> pd.DataFrame:
        """
        Get the features from the DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing the features, or an empty DataFrame if no data is loaded.
        """
        if self.__data is None:
            raise ValueError("No data loaded")

        if self.__label_column not in self.__data.columns:
            raise ValueError("Label column not found")

        return self.__data.drop(columns=[self.__label_column], errors="ignore")

    def get_labels(self) -> pd.Series:
        """
        Get the labels from the DataFrame.

        Returns:
            pd.Series: Series containing the labels, or an empty Series if no data is loaded.
        """
        if self.__data is None:
            raise ValueError("No data loaded")

        if self.__label_column not in self.__data.columns:
            raise ValueError("Label column not found")

        return (
            self.__data[self.__label_column]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    def split_data(self, test_size: float = 0.3, random_state: int = 39) -> tuple:
        """
        Split the data into training and testing sets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: Features and labels for training and testing sets.
        """

        # Ensure basic validity of the dataset
        self.clean_data()

        if self.__label_column not in self.__data.columns:
            raise ValueError("Label column not found")

        X = self.get_features()
        y = self.get_labels()
        if y.nunique() < 2:
            raise ValueError(
                "Label column contains fewer than 2 unique classes after cleaning."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, X_test, y_train, y_test










