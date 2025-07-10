from services.input_validator import DataValidator
from services.cleaner_and_spliter import Cleaner
from src.train import NaiveBayesTrainer
from src.evaluator import NaiveBayesEvaluator

class AppController:
    def __init__(self, label_col, loader):
        self.label_col = label_col
        self.loader = loader

        self.data = None
        self.label_name = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.classifier = None
        self.evaluator = None

    def load_and_prepare(self, file_path):
        file_path, self.label_name = DataValidator.validate_cli(file_path, self.label_col)
        self.data = self.loader.load_data(self.file_path)
        # print(f"[DEBUG] Loaded data:\n{self.data.head()}")
        cleaner = Cleaner(self.data, self.label_name)
        self.X_train, self.X_test, self.y_train, self.y_test = cleaner.split_data()

    def train_model(self):
        self.classifier = NaiveBayesTrainer()
        model = self.classifier.fit(self.X_train, self.y_train)
        self.evaluator = NaiveBayesEvaluator(model)

    def get_accuracy(self):
        if not self.evaluator:
            raise Exception("Model not trained.")
        acc = self.evaluator.evaluate(self.X_test, self.y_test)
        return acc

    def get_schema(self):
        schema = {}
        for col in self.X_train.columns:
            options = self.X_train[col].astype(str).unique().tolist()
            schema[col] = options
        return schema

    def predict_record(self, record: dict):
        if not self.evaluator:
            raise Exception("Model not trained.")
        return self.evaluator.predict(record)
