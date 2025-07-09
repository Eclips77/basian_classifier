from abc import ABC,abstractmethod

class DataLoader:

    @abstractmethod
    def load_data(file_path: str):
        pass




