import pandas as pd

class Classifier:

    def __init__(self):
        self.__classes = []
        self.__class_priors = {}
        self.__conditional_probs = {}


    def fit(self, featureData : pd.DataFrame, labels : pd.Series):
        """
        Train the Naive Bayes Classifier using the training data.

        Args:
            FeatureData (pd.DataFrame): The feature data.
            labels (pd.Series): The label data.
        """
        # Get unique classes
        self.__classes = list(labels.unique())

        # Compute prior probabilities for each class
        class_counts = labels.value_counts().to_dict()
        total_samples = len(labels)
        self.__class_priors = {}

        for cls in self.__classes:
            self.__class_priors[cls] = class_counts[cls] / total_samples

        # Compute conditional probabilities with Laplacian correction
        self.__conditional_probs = {}

        for feature in featureData.columns:
            self.__conditional_probs[feature] = {}
            feature_values = featureData[feature].unique()

            for value in feature_values:
                value = str(value).strip()
                self.__conditional_probs[feature][value] = {}

                for cls in self.__classes:
                    # Count where feature == value and class == cls
                    count = len(featureData[(featureData[feature] == value) & (labels == cls)])
                    total_in_class = class_counts[cls]
                    num_possible_values = len(feature_values)
                    
                    # Laplacian Correction
                    prob = (count + 1) / (total_in_class + num_possible_values)
                    
                    self.__conditional_probs[feature][value][cls] = prob

    def predict(self, record: dict):
        """
        Predict the class label for a single record.

        Args:
            record (dict): The input record as a dictionary.

        Returns:
            str: The predicted class label.
        """
        class_probs = {}

        for cls in self.__classes:
            # Start with the prior probability
            prob = self.__class_priors[cls]

            for feature, value in record.items():
                value = str(value).strip()
                if feature in self.__conditional_probs:
                    if value in self.__conditional_probs[feature]:
                        prob *= self.__conditional_probs[feature][value].get(cls, 1e-6)
                    else:
                        prob *= 1e-6  # Smoothing for unseen value
                else:
                    prob *= 1e-6  # Smoothing for unseen feature

            class_probs[cls] = prob
    # Return the class with the highest probability
        return max(class_probs, key=class_probs.get)

    def predict_batch(self, X: pd.DataFrame):
        """
        Predict class labels for a batch of records.

        Args:
            X (pd.DataFrame): The input features data.

        Returns:
            list: List of predicted class labels.
        """
        predictions = []
        for _, row in X.iterrows():
            record = row.to_dict()
            pred = self.predict(record)
            predictions.append(pred)
        return predictions


    


















