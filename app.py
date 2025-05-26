from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# Load the trained model and scaler
model = joblib.load('loan_status_predictor.pkl')
scaler = joblib.load('vector.pkl')

# List of numerical columns to scale
num_cols = [
    'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
    'residential_assets_value', 'commercial_assets_value',
    'luxury_assets_value', 'bank_asset_value'
]

# Initialize FastAPI app
app = FastAPI(title="Loan Status Prediction API")

# Define input schema using Pydantic
class LoanApproval(BaseModel):
    no_of_dependents: float
    education: float
    self_employed: float
    income_annum: float
    loan_amount: float
    loan_term: float
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

@app.post("/predicts", summary="Predict Loan Approval Status")
async def predict_loan_status(application: LoanApproval):
    """Predict loan approval based on applicant financial data."""
    # Convert input to DataFrame
    input_data = pd.DataFrame([application.dict()])

    # Apply scaling to numerical columns
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # Make prediction
    prediction = model.predict(input_data)

    # Return result
    status = "Approved" if prediction[0] == 1 else "Not Approved"
    return {"Loan Status": status}
