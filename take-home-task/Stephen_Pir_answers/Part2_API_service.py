import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager

# --- 0. Setup and Configuration ---

# Get the directory of the current script to build robust file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_ROOT, "artifacts", "model.joblib")

# --- 1. Load Model ---

# Global variable to hold the model
model: Optional[object] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Context manager to load the ML model at startup and clean up at shutdown.
    This is the recommended modern approach for startup/shutdown events.
    """
    global model
    print("--- Starting up and loading model ---")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
        print("The '/predict' endpoint will be unavailable.")
        model = None
    except Exception as e:
        # More robust error handling for investigating joblib loading issues. 
        error_type = type(e).__name__
        print(f"ERROR: An unexpected error of type '{error_type}' occurred while loading the model.")
        print(f"       Details: {e}")
        print("       This can happen if the model file is corrupted or not a valid joblib file.")
        print("The '/predict' endpoint will be unavailable.")
        model = None
    
    yield
    
    # --- Shutdown logic (if any) ---
    print("--- Shutting down ---")
    model = None

# --- 2. FastAPI App Initialization ---

app = FastAPI(
    title="Credit Default Prediction API",
    description="A simple API to predict credit default risk based on customer transaction features.",
    version="1.0.0",
    lifespan=lifespan
)

# --- 3. Pydantic Models for Input and Output ---

# Based on the training_set.csv from Part 1.
# This model reflects the full feature set but does not fit the provided library model, so included for information purposes only.
# class CustomerFeatures(BaseModel):
#     customer_id: str
#     num_transactions: int
#     avg_transaction_amount: float
#     total_debit: float
#     total_credit: float
#     balance_change: float
#     debit_credit_ratio: float
#     std_dev_amount: float
#     has_rent: int
#     has_payroll: int
#     has_netflix: int
#     has_tesco: int
#     has_bonus: int
#     defaulted_within_90d: int # This would not be included in input for prediction

# However, the provided model expects a reduced feature set as per the task description.
class CustomerFeatures(BaseModel):
    customer_id: str
    txn_count: float
    total_debit: float
    total_credit: float
    avg_amount: float
    kw_rent: int = 0
    kw_netflix: int = 0
    kw_tesco: int = 0
    kw_payroll: int = 0
    kw_bonus: int = 0
    

class PredictionResponse(BaseModel):
    customer_id: str
    probability: float
    prediction: int

# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the Credit Default Prediction API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: CustomerFeatures):
    """
    Accepts customer features and returns a default prediction.
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model is not loaded. The service is unavailable."
        )

    # This reflects the full feature set but is not what is expected by the provided model.
    # Only for information purposes.
    # feature_order = [
    #      'customer_id', 
    #      'num_transactions',
    #      'avg_transaction_amount',
    #      'total_debit',
    #      'total_credit',
    #      'balance_change',
    #      'debit_credit_ratio',
    #      'std_dev_amount',
    #      'has_rent',
    #      'has_payroll',
    #      'has_netflix', 
    #      'has_tesco',
    #      'has_bonus'
    # ]

    # The feature order must match the order used during model training.
    feature_order = [
            'txn_count', 'total_debit', 'total_credit', 'avg_amount', 
        'kw_rent', 'kw_netflix', 'kw_tesco', 'kw_payroll', 'kw_bonus'
    ]
    # Create a DataFrame from the input features
    # .model_dump() replaces the deprecated .dict() method in Pydantic v2
    input_data_dict = features.model_dump()
    input_df = pd.DataFrame([input_data_dict])[feature_order]

    # Get probability of the positive class (class 1)
    probability = model.predict_proba(input_df)[0, 1]
    prediction = model.predict(input_df)[0]

    return PredictionResponse(
        customer_id=features.customer_id,
        # probability=probability,
        probability=round(probability, 4), # Round for cleaner output, chose 4dp although not specified in requirement and exemplar was 2dp
        prediction=int(prediction)
    )