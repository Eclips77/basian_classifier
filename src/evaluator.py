import math
import pandas as pd

class NaiveBayesEvaluator:
    def __init__(self, model):
        self.model = model

    def _log_conditional(self, feature, value, cls):
        value = str(value).strip().lower()
        count = self.model.feature_value_counts[feature][cls].get(value, 0)
        denom = self.model.class_counts[cls] + self.model.feature_possible_values[feature]
        return math.log((count + 1) / denom)

    def predict(self, record: dict):
        log_probs = {}
        for cls in self.model.classes:
            log_prob = self.model.priors[cls]
            for feature, value in record.items():
                log_prob += self._log_conditional(feature, value, cls)
            log_probs[cls] = log_prob
        best_class = max(log_probs, key=log_probs.get)
        print(f"predict : {best_class} with probability {log_probs}")
        return best_class

    def predict_batch(self, X: pd.DataFrame):
        return X.apply(lambda row: self.predict(row.to_dict()), axis=1)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        predictions = self.predict_batch(X_test)
        correct = sum(pred == true for pred, true in zip(predictions, y_test))
        accuracy = correct / len(y_test)
        print(f"[Evaluator] accuracy :{accuracy:.2%}")
        return accuracy
