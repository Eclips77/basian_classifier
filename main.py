from src.classifier import Classifier
from services.data_loader import DataLoader

# שים לב - זה ברוט, אז הנתיבים נכונים
loader = DataLoader("Data/buy_computer.csv", "BoughtComputer")

# חשוב! טוען דאטה לפני שמוציא תכונות
loader.load_data()
loader.clean_data()

x = loader.get_features()
y = loader.get_labels()

clf = Classifier()
clf.fit(x,y)

# הדפס מה יצא
print("\n=== Class Priors ===")
print(clf._Classifier__class_priors)

print("\n=== Conditional Probabilities ===")
for feature, value_dict in clf._Classifier__conditional_probs.items():
    print(f"Feature: {feature}")
    for value, probs in value_dict.items():
        print(f"  Value: {value} => {probs}")
