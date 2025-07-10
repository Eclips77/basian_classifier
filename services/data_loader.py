from abc import ABC,abstractmethod

class DataLoader:

    @abstractmethod
    def load_data(self,file_path: str):
        pass




