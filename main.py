from services.data_loader import DataLoader
from services.input_validator import DataValidator
from src.classifier import Classifier
from src.record_classifier import RecordClassifier

FILE_PATH = "Data/agaricus-lepiota.csv"
LABEL_COL = 0

path, label = DataValidator.validate_cli(FILE_PATH, LABEL_COL)

loader = DataLoader(path, label)
loader.load_data()
loader.clean_data()

X_train, X_test, y_train, y_test = loader.split_data(test_size=0.3)

clf = Classifier()
clf.fit(X_train, y_train)

rc = RecordClassifier(clf)
accuracy = rc.evaluate(X_test, y_test)
print(f"\n=== FINAL ACCURACY === {accuracy:.2%}\n")

# for idx, row in X_test.iterrows():
#     record = row.to_dict()
#     pred = clf.predict(record)
#     log_probs = {
#         cls: clf.priors[cls]
#         + sum(clf._log_conditional(f, record[f], cls) for f in X_test.columns)
#         for cls in clf.classes
#     }
#     print(f"#{idx} true='{y_test.loc[idx]}' pred='{pred}' log-probs={log_probs}")
