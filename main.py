from services.data_loader import DataLoader
from services.input_validator import DataValidator
from src.classifier import Classifier
from src.record_classifier import RecordClassifier

file_path = "Data/buy_computer.csv"
label_column = "BoughtComputer"  

validated_path, validated_label = DataValidator.validate_cli(file_path, label_column)

# 2. Load and clean data
loader = DataLoader(validated_path, validated_label)
loader.load_data()
loader.clean_data()

# 3. Split data into training and testing sets using DataLoader's method
X_train, X_test, y_train, y_test = loader.split_data(test_size=0.3)

# 4. Train the classifier
clf = Classifier()
clf.fit(X_train, y_train)

# 5. Evaluate using RecordClassifier
rc = RecordClassifier(clf)
accuracy = rc.evaluate(X_test, y_test)

print(f"\n=== Model Accuracy ===\nAccuracy: {accuracy:.2%}")
