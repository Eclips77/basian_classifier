from src.classifier import Classifier
from services.data_loader import DataLoader
from services.input_validator import DataValidator

path = "Data/buy_computer.csv"
label_column = 0

validated_filepath, validated_label = DataValidator.validate_cli(path, label_column)

loader = DataLoader(validated_filepath,validated_label )

loader.load_data()
loader.clean_data()

x = loader.get_features()
y = loader.get_labels()

clf = Classifier()
clf.fit(x,y)

print("\n=== Class Priors ===")
print(clf._Classifier__class_priors)

print("\n=== Conditional Probabilities ===")
for feature, value_dict in clf._Classifier__conditional_probs.items():
    print(f"Feature: {feature}")
    for value, probs in value_dict.items():
        print(f"  Value: {value} => {probs}")
