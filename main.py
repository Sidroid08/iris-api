from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from typing import List

app = FastAPI(
    title="Iris Classifier API",
    description="API for classifying iris flowers based on measurements",
    version="1.0.0"
)

# Load model and scaler with absolute paths
def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'saved_models/iris_model.pkl')
        scaler_path = os.path.join(os.path.dirname(__file__), 'saved_models/iris_scaler.pkl')
        return joblib.load(model_path), joblib.load(scaler_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model files: {str(e)}")

model, scaler = load_model()

# Feature names from Iris dataset
IRIS_FEATURES = [
    "sepal_length (cm)",
    "sepal_width (cm)",
    "petal_length (cm)",
    "petal_width (cm)"
]

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Iris Classifier API!",
        "endpoints": {
            "info": "/info (GET)",
            "predict": "/predict (POST)"
        },
        "documentation": "/docs or /redoc"
    }

@app.get("/info")
async def get_feature_info():
    """Returns the list of features required for prediction"""
    return {
        "feature_names": IRIS_FEATURES,
        "description": "All measurements should be in centimeters",
        "expected_order": IRIS_FEATURES,
        "sample_input": {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
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
        probabilities = model.predict_proba(scaled_input)[0]
        
        species = ["setosa", "versicolor", "virginica"]
        predicted_species = species[prediction[0]]
        
        return {
            "prediction": predicted_species,
            "probabilities": {
                "setosa": float(probabilities[0]),
                "versicolor": float(probabilities[1]),
                "virginica": float(probabilities[2])
            },
            "input_features": iris_data.dict()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
        )

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
