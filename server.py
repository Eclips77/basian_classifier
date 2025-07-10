from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.data_loader import DataLoader
from src.app_controller import AppController

app = FastAPI()

loader = DataLoader()
controller = AppController(label_col="BoughtComputer", loader=loader)

controller.load_and_prepare("Data/buy_computer.csv")
controller.train_model()

class Record(BaseModel):
    record: dict


@app.get("/accuracy")
def get_accuracy():
    try:
        acc = controller.get_accuracy()
        return {"accuracy": acc}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/schema")
def get_schema():
    try:
        return controller.get_schema()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict(record: Record):
    try:
        result = controller.predict_record(record.record)
        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


