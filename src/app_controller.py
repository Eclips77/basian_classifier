from services.data_loader import DataLoader
from services.input_validator import DataValidator
from src.train import NaiveBayesTrainer
from src.evaluator import NaiveBayesEvaluator
from services.cleaner_and_spliter import Cleaner


class AppController:
    def __init__(self, file_path: str, label_col: str, loader: DataLoader):
        self.file_path = file_path
        self.label_col = label_col
        self.loader = loader

        self.classifier = NaiveBayesTrainer()
        self.evaluator = None

        self.data = None
        self.label_name = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def run(self):
        print(" Welcome to the naive basian app!")

        while True:
            print("\n--- MENU ---")
            print("1. Load and clean data")
            print("2. Train model")
            print("3. Evaluate model accuracy")
            print("4. Predict a single record")
            print("0. Exit")

            choice = input("Select an option: ")

            if choice == "1":
                self.load_and_prepare()
            elif choice == "2":
                self.train_model()
            elif choice == "3":
                self.evaluate_model()
            elif choice == "4":
                self.predict_record()
            elif choice == "0":
                print("[AppController] Exiting. Goodbye!")
                break
            else:
                print("[AppController] Invalid choice.")

    def load_and_prepare(self):
        self.file_path, self.label_name = DataValidator.validate_cli(self.file_path, self.label_col)
        self.data = self.loader.load_data(self.file_path)

        cleaner = Cleaner(self.data, self.label_name)
        self.X_train, self.X_test, self.y_train, self.y_test = cleaner.split_data()
        print("Data loaded and cleaned successfully.")

    def train_model(self):
        if self.X_train is None:
            print("Please load and clean data first.")
            return
        model = self.classifier.fit(self.X_train, self.y_train)
        self.evaluator = NaiveBayesEvaluator(model)
        print("Model trained successfully.")

    def evaluate_model(self):
        if self.evaluator is None:
            print("Please train the model first.")
            return
        self.evaluator.evaluate(self.X_test, self.y_test)

    def predict_record(self):
        if self.evaluator is None:
            print("Please train the model first.")
            return
        record = {}
        for col in self.X_test.columns:
            while True:
                val = input(f"Enter value for '{col}': ").strip()
                if val:
                    break
                print(f"Invalid input for '{col}'. Please try again.")

            record[col] = val
        result = self.evaluator.predict(record)
        print(f"Prediction: {result}")
