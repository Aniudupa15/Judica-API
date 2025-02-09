from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

router = APIRouter()

# Define paths for preprocessing objects and model
preprocessing_path = os.path.join("models", "preprocessing_objects.pkl")
model_path = os.path.join("models", "bail_reckoner_model.pkl")

# Load preprocessing objects and model with error handling
try:
    preprocessing_objects = joblib.load(preprocessing_path)
    if preprocessing_objects is None:
        raise FileNotFoundError(f"Preprocessing objects file is empty or corrupted: {preprocessing_path}")
    
    label_encoders = preprocessing_objects.get('label_encoders', {})
    scaler = preprocessing_objects.get('scaler', None)
    if not label_encoders:
        raise KeyError("Label encoders are missing from the preprocessing objects.")
    if not scaler:
        raise KeyError("Scaler object is missing from the preprocessing objects.")
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=str(e))
except KeyError as e:
    raise HTTPException(status_code=500, detail=f"Missing key in preprocessing objects: {str(e)}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading preprocessing objects: {str(e)}")

# Load the bail reckoner model
try:
    model = joblib.load(model_path)
    if model is None:
        raise FileNotFoundError(f"Model file is empty or corrupted: {model_path}")
except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

# Define Pydantic model for input data
class BailInput(BaseModel):
    statute: str
    offense_category: str
    penalty: str
    imprisonment_duration_served: int
    risk_of_escape: int
    risk_of_influence: int
    surety_bond_required: int
    personal_bond_required: int
    fines_applicable: int
    served_half_term: int
    risk_score: float
    penalty_severity: float

@router.post("/predict-bail")
async def predict_bail(data: BailInput):
    try:
        # Convert input data to DataFrame for model prediction
        user_input = pd.DataFrame([data.dict()])
        
        # Apply label encoding to categorical columns
        for col, encoder in label_encoders.items():
            if col in user_input:
                user_input[col] = encoder.transform(user_input[col])
        
        # Scale the numerical columns
        numerical_columns = ['imprisonment_duration_served', 'risk_score', 'penalty_severity']
        user_input[numerical_columns] = scaler.transform(user_input[numerical_columns])
        
        # Make the prediction
        result = model.predict(user_input)
        prediction = "Eligible for Bail" if result[0] == 1 else "Not Eligible for Bail"
        
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing prediction request: {str(e)}")

@router.get("/")
async def root():
    return {"message": "Bail Reckoner API is running."}
