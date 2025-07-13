from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from services.file_loader import FileLoader
from src.app_controller import AppController

app = FastAPI()

loader = FileLoader()
controller = None
loaded_file_path = None

class Record(BaseModel):
    record: dict


class LoadRequest(BaseModel):
    file_path: str


class TrainRequest(BaseModel):
    file_path: str | None = None
    label_column: str


@app.post("/load_file")
def load_file(req: LoadRequest):
    """Load a CSV file and return its column names."""
    global loaded_file_path
    try:
        df = loader.load_data(req.file_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=400, detail=str(exc))
    loaded_file_path = req.file_path
    return {"columns": df.columns.tolist()}


@app.post("/train")
def train(req: TrainRequest):
    """Load data and train a model based on the chosen label column."""
    global controller, loaded_file_path
    file_path = req.file_path or loaded_file_path
    if not file_path:
        raise HTTPException(status_code=400, detail="No file selected")

    try:
        controller = AppController(label_col=req.label_column, loader=loader)
        controller.load_and_prepare(file_path)
        controller.train_model()
        acc = controller.get_accuracy()
        return {"accuracy": acc}
    except Exception as exc:  # pylint: disable=broad-except
        controller = None
        raise HTTPException(status_code=400, detail=str(exc))


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


