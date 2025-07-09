class NaiveBayesModel:
    def __init__(self, classes, class_counts, priors, feature_value_counts, feature_possible_values):
        self.classes = classes
        self.class_counts = class_counts
        self.priors = priors
        self.feature_value_counts = feature_value_counts
        self.feature_possible_values = feature_possible_values
