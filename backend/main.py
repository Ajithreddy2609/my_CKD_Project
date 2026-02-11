from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import traceback

app = FastAPI(title="CKD Clinical Decision Support API")

# --- 1. HEALTH CHECK ---
@app.get("/")
def read_root():
    return {"status": "CKD API is running", "version": "1.1.0"}

# --- 2. CONFIGURATION ---
MODEL_FILE = "backend/rf_pipeline.pkl" 

MEDICAL_RANGES = {
    "bp": {"min": 60, "max": 120, "unit": "mm/Hg", "label": "Blood Pressure"},
    "sg": {"min": 1.005, "max": 1.030, "unit": "", "label": "Specific Gravity"},
    "hemo": {"min": 13.5, "max": 17.5, "unit": "g/dL", "label": "Hemoglobin"},
    "bgr": {"min": 70, "max": 140, "unit": "mg/dL", "label": "Blood Glucose"},
    "bu": {"min": 7, "max": 20, "unit": "mg/dL", "label": "Blood Urea"},
    "sc": {"min": 0.6, "max": 1.2, "unit": "mg/dL", "label": "Serum Creatinine"},
    "sod": {"min": 135, "max": 145, "unit": "mEq/L", "label": "Sodium"},
    "pot": {"min": 3.5, "max": 5.1, "unit": "mEq/L", "label": "Potassium"},
    "wc": {"min": 4500, "max": 11000, "unit": "cells/mcL", "label": "White Cell Count"},
    "rc": {"min": 4.5, "max": 5.9, "unit": "million/mcL", "label": "Red Cell Count"}
}

# --- 3. MIDDLEWARE & MODEL LOAD ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    with open(MODEL_FILE, "rb") as file:
        pipeline = pickle.load(file)
except FileNotFoundError:
    print(f"CRITICAL ERROR: {MODEL_FILE} not found.")
    pipeline = None

# --- 4. DATA MODELS ---
class CKDInput(BaseModel):
    age: float; bp: float; sg: float; al: float; su: float
    bgr: float; bu: float; sc: float; sod: float; pot: float
    hemo: float; pcv: float; wc: float; rc: float
    rbc: str; pc: str; pcc: str; ba: str; htn: str
    dm: str; cad: str; appet: str; pe: str; ane: str

# --- 5. PREDICTION ENDPOINT ---
@app.post("/predict")
async def predict_ckd(data: CKDInput):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model pipeline not found.")

    try:
        user_dict = data.model_dump()
        input_df = pd.DataFrame([user_dict])
        
        feature_order = [
            "age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "wc", "rc",
            "rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"
        ]
        input_df = input_df[feature_order]

        # 1. Get Probability
        probs = pipeline.predict_proba(input_df)[0]
        probability_val = probs[1] if len(probs) > 1 else probs[0]

        # 2. HIGH CERTAINTY THRESHOLD (Set to 70%)
        # This requires the AI to be very sure before flagging Positive
        is_positive = (probability_val > 0.70) 

        # 3. Synchronized Clinical Insights
        insights = []
        for key, range_info in MEDICAL_RANGES.items():
            val = user_dict.get(key)
            
            # Logic: Tighten the 'Normal' window only if AI detected high risk
            buffer = 0.05 if is_positive else 0
            lower_bound = range_info["min"] * (1 + buffer)
            upper_bound = range_info["max"] * (1 - buffer)

            if key == "sg":
                status = "Normal" if val >= lower_bound else "At Risk"
            else:
                status = "Normal" if (range_info["min"] <= val <= range_info["max"]) else "Abnormal"
            
            # Dynamic Labeling for the B.Tech Presentation Logic
            if is_positive and status == "Normal":
                if val >= range_info["max"] * 0.9 or val <= range_info["min"] * 1.1:
                    status = "Borderline"

            insights.append({
                "parameter": range_info["label"],
                "value": val,
                "range": f"{range_info['min']} - {range_info['max']} {range_info['unit']}",
                "status": status
            })

        return {
            "prediction": "Positive for CKD" if is_positive else "Negative for CKD",
            "probability": f"{round(probability_val * 100, 2)}%",
            "insights": insights,
            "recommendation": "High Risk Detected. Consult a nephrologist." if is_positive else "Everything appears normal. Maintain regular checkups.",
            "disclaimer": "Academic Prototype: Not a clinical diagnosis."
        }
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))