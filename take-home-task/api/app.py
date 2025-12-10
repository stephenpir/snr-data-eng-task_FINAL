from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

app = FastAPI(title="ML Inference Service")

class CustomerFeatures(BaseModel):
    txn_count: float
    total_debit: float
    total_credit: float
    avg_amount: float
    kw_rent: int = 0
    kw_netflix: int = 0
    kw_tesco: int = 0
    kw_payroll: int = 0
    kw_bonus: int = 0

model = None

@app.on_event("startup")
def load_model():
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Please place model.joblib in artifacts/")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    X = [[
        payload.txn_count,
        payload.total_debit,
        payload.total_credit,
        payload.avg_amount,
        payload.kw_rent,
        payload.kw_netflix,
        payload.kw_tesco,
        payload.kw_payroll,
        payload.kw_bonus,
    ]]
    proba = model.predict_proba(X)[0][1]
    pred = int(proba >= 0.5)
    return {"probability": proba, "prediction": pred}