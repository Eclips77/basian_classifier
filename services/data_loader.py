"""Base interface for loading data."""

from abc import ABC, abstractmethod


class DataLoader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load_data(self, file_path: str):
        """Load data from ``file_path`` and return it.

        Implementations should raise ``ValueError`` with a helpful message when
        loading fails so calling code can present a clear error to the user.
        """
        raise NotImplementedError





