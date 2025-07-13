from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from services.file_loader import FileLoader
from src.app_controller import AppController

app = FastAPI()

loader = FileLoader()
controller = AppController(label_col="BoughtComputer", loader=loader)

DATA_PATH = os.getenv("DATA_PATH") or os.path.join(FileLoader.DEFAULT_DATA_DIR, "buy_computer.csv")

try:
    controller.load_and_prepare(DATA_PATH)
    controller.train_model()
except Exception as exc:  # pylint: disable=broad-except
    # If the data fails to load, keep the server running but warn the user.
    print(f"Failed to initialise data from {DATA_PATH}: {exc}")
    controller = None

class Record(BaseModel):
    record: dict


@app.get("/accuracy")
def get_accuracy():
    if controller is None:
        raise HTTPException(status_code=400, detail="Data not initialised")
    try:
        acc = controller.get_accuracy()
        return {"accuracy": acc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/schema")
def get_schema():
    if controller is None:
        raise HTTPException(status_code=400, detail="Data not initialised")
    try:
        return controller.get_schema()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(record: Record):
    if controller is None:
        raise HTTPException(status_code=400, detail="Data not initialised")
    try:
        result = controller.predict_record(record.record)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


