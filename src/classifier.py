import math
import pandas as pd


class Classifier:
    def __init__(self):
        self.classes = []
        self.class_counts = {}
        self.priors = {}
        self.feature_value_counts = {}
        self.feature_possible_values = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes = list(y.unique())
        self.class_counts = y.value_counts().to_dict()
        total = len(y)
        self.priors = {cls: math.log(cnt / total) for cls, cnt in self.class_counts.items()}

        for feature in X.columns:
            self.feature_value_counts[feature] = {}
            self.feature_possible_values[feature] = len(X[feature].astype(str).unique())
            for cls in self.classes:
                subset = X[y == cls][feature].astype(str).str.strip().str.lower()
                counts = subset.value_counts().to_dict()
                self.feature_value_counts[feature][cls] = counts

    def _log_conditional(self, feature, value, cls):
        value = str(value).strip().lower()
        count = self.feature_value_counts[feature][cls].get(value, 0)
        denom = self.class_counts[cls] + self.feature_possible_values[feature]
        return math.log((count + 1) / denom)

    def predict(self, record: dict):
        log_probs = {}
        for cls in self.classes:
            log_prob = self.priors[cls]
            for feature, value in record.items():
                log_prob += self._log_conditional(feature, value, cls)
            log_probs[cls] = log_prob
        return max(log_probs, key=log_probs.get)

    def predict_batch(self, X: pd.DataFrame):
        return [self.predict(row.to_dict()) for _, row in X.iterrows()]
