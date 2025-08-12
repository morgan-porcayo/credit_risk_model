#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 13:25:27 2025

@author: morganporcayo
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained pipeline
model = joblib.load("credit_model.joblib")

# Define the request body structure
class ApplicantData(BaseModel):
    person_age: int
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str
        
# Initialize FastAPI app
app = FastAPI(title="Credit Approval API", version="1.0")

@app.post("/predict")
def predict_credit(data: ApplicantData):
    # Convert incoming data to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]
    
    return {
        "approved": bool(prediction),
        "approval_probability": round(float(probability), 4)
    }

# Health check endpoint
@app.get("/")
def read_root():
    return {"message": "Credit Approval API is running!"}
