from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI()

# Load model and scaler
model = joblib.load("saved_models/iris_model.pkl")
scaler = joblib.load("saved_models/iris_scaler.pkl")

# Feature names from Iris dataset
IRIS_FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Iris Classifier API!",
        "usage": "Send a POST request to /predict with flower measurements"
    }

@app.get("/info")
async def get_feature_info():
    """Returns the list of features required for prediction"""
    return {
        "feature_names": IRIS_FEATURES,
        "description": "All measurements should be in centimeters",
        "expected_order": IRIS_FEATURES
    }

@app.post("/predict")
async def predict(iris_data: IrisInput):
    try:
        # Convert input to array and validate
        input_array = np.array([
            iris_data.sepal_length,
            iris_data.sepal_width,
            iris_data.petal_length,
            iris_data.petal_width
        ]).reshape(1, -1)
        
        # Check for NaN/infinity
        if not np.isfinite(input_array).all():
            raise HTTPException(status_code=400, detail="Input contains invalid values")
            
        # Scale and predict
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        species = ["setosa", "versicolor", "virginica"]
        
        return {"prediction": species[prediction[0]]}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))