import pandas as pd


class RecordClassifier:
    def __init__(self, classifier):
        self.classifier = classifier

    def classify_record(self, record: dict):
        return self.classifier.predict(record)

    def classify_batch(self, X: pd.DataFrame):
        return self.classifier.predict_batch(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> float:
        y_test = y_test.astype(str).str.strip().str.lower()
        predictions = self.classify_batch(X_test)
        correct = sum(predicted_label.lower() == true for predicted_label, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        print(f"Model accuracy: {accuracy:.2%}")
        return accuracy
