from src.app_controller import AppController
from services.file_loader import FileLoader

FILE_PATH = "Data/phishing.csv"
LABEL_COL = "class"

# app = AppController(FILE_PATH, LABEL_COL)
# app.run()


if __name__ == "__main__":
    loader = FileLoader()
    app = AppController(FILE_PATH, LABEL_COL, loader=loader)
    app.run()



# for idx, row in X_test.iterrows():
#     record = row.to_dict()
#     pred = clf.predict(record)
#     log_probs = {
#         cls: clf.priors[cls]
#         + sum(clf._log_conditional(f, record[f], cls) for f in X_test.columns)
#         for cls in clf.classes
#     }
#     print(f"#{idx} true='{y_test.loc[idx]}' pred='{pred}' log-probs={log_probs}")
