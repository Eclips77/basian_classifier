import math
import pandas as pd
from classifier_model import NaiveBayesModel

class NaiveBayesTrainer:
    def fit(self, X: pd.DataFrame, y: pd.Series) -> NaiveBayesModel:
        print("[Trainer] start training...")
        classes = list(y.unique())
        class_counts = y.value_counts().to_dict()
        total = len(y)
        priors = {cls: math.log(cnt / total) for cls, cnt in class_counts.items()}

        feature_value_counts = {}
        feature_possible_values = {}

        for feature in X.columns:
            feature_value_counts[feature] = {}
            feature_possible_values[feature] = len(X[feature].astype(str).unique())

            for cls in classes:
                subset = X[y == cls][feature].astype(str).str.strip().str.lower()
                counts = subset.value_counts().to_dict()
                feature_value_counts[feature][cls] = counts

        print("[Trainer] stop trining")
        return NaiveBayesModel(classes, class_counts, priors, feature_value_counts, feature_possible_values)
