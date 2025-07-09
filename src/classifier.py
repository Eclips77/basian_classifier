import pandas as pd

class Classifier:

    def __init__(self):
        self.classes = []
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
                self.__conditional_probs[feature][value] = {}

                for cls in self.__classes:
                    # Count where feature == value and class == cls
                    count = len(featureData[(featureData[feature] == value) & (labels == cls)])
                    total_in_class = class_counts[cls]
                    num_possible_values = len(feature_values)
                    
                    # Laplacian Correction
                    prob = (count + 1) / (total_in_class + num_possible_values)
                    
                    self.__conditional_probs[feature][value][cls] = prob


    def predict(self, record):
        """
        Predict the class labels for the input data.
        """
        pass

    def predict_batch(self, x):
        """
        Predict the class labels for a batch of input data.
        """
        pass

    def evaluate(self, test_X, test_y):
        """
        Evaluate the classifier on the test data.
        """
        pass


















