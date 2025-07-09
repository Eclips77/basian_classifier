class RecordClassifier:
    def __init__(self, classifier):
        """
        Initialize with a trained Classifier.
        """
        self.classifier = classifier

    def classify_record(self, record):
        """
        Classify a single record using the trained classifier.

        Args:
            record (dict): The input record.

        Returns:
            str: Predicted class label.
        """
        return self.classifier.predict(record)

    def classify_batch(self, X):
        """
        Classify a batch of records using the trained classifier.

        Args:
            X (pd.DataFrame): The input features.

        Returns:
            list: Predicted class labels.
        """
        return self.classifier.predict_batch(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model accuracy on test data.

        Args:
            X_test (pd.DataFrame): The test features.
            y_test (pd.Series): The true labels.

        Returns:
            float: The accuracy score as a decimal.
        """
        predictions = self.classify_batch(X_test)
        correct = 0

        for pred, true_label in zip(predictions, y_test):
            if pred == true_label:
                correct += 1

        accuracy = correct / len(y_test)
        print(f"Model accuracy: {accuracy:.2%}")
        return accuracy
