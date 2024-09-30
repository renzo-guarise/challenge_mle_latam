import fastapi
import pickle
import pandas as pd
from pydantic import BaseModel
from typing import List, Dict
from fastapi import HTTPException
from .model import *

app = fastapi.FastAPI()
model_lr = None

try:
    # Load the model once during the startup of the FastAPI app
    with open('model_v1.bin', 'rb') as f_in:
        model_lr = pickle.load(f_in)
    print("Model loaded successfully.")
except FileNotFoundError:
    raise print("File not found. Please check the file path.")
except PermissionError:
    print("Permission denied. Please check your file permissions.")
except pickle.UnpicklingError:
    print("Unpickling error. The file might be corrupted or incompatible.")
except Exception as e:
    print(f"An unexpected error occurred while loading the model: {e}")

# Define a schema for the input data using Pydantic
class PredictionInput(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int


# Health check endpoint
@app.get("/health", status_code=200)
async def get_health() -> Dict[str, str]:
    return {
        "status": "OK"
    }

# Enhanced Prediction endpoint with error handling
@app.post("/predict", status_code=200)
async def post_predict(input_data: Dict[str,List[PredictionInput]]) -> Dict[str, List]:
    if model_lr is None:
        raise HTTPException(status_code=500, detail="Model not loaded successfully.")

    try:
        # Convert input data to DataFrame
        data = pd.DataFrame([item.dict() for item in input_data["flights"]])
        print(data)
        
        # Preprocess the data
        features = model_lr.preprocess(data)
        
        # Make predictions
        predictions = model_lr._model.predict(features)
        
        # Return predictions as a list
        return {"predict": predictions.tolist()}
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Input data unseen: {e}")
    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        # Catch any other unforeseen exceptions
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")