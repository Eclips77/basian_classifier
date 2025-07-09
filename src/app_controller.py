from services.data_loader import DataLoader
from services.input_validator import DataValidator
from src.classifier import Classifier
from src.record_classifier import RecordClassifier
from services.cleaner_and_spliter import Cleaner

class AppController:
    def __init__(self, file_path: str, label_col: str):
        self.file_path = file_path
        self.label_col = label_col
        self.loader = DataLoader()
        self.clf = Classifier()
        self.rc = RecordClassifier(self.clf)

    def run(self):
        path, label = DataValidator.validate_cli(self.file_path, self.label_col)
        self.loader.load_data(path)
        data = self.loader.get_data()
        
        cleaner = Cleaner(data, label)
        X_train, X_test, y_train, y_test = cleaner.split_data()

        self.clf.fit(X_train, y_train)
        self.rc.evaluate(X_test, y_test)