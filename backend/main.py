from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import joblib
import json
import os
import io

from src.features import make_features, LABELS, COST_MATRIX

app = FastAPI(title="Triage API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load artifacts
artifacts_dir = os.path.join("outputs", "artifacts")
try:
    models = joblib.load(os.path.join(artifacts_dir, 'models.joblib'))
    calibrator = joblib.load(os.path.join(artifacts_dir, 'calibrator.joblib'))
    feature_cols = joblib.load(os.path.join(artifacts_dir, 'feature_cols.joblib'))
    
    with open(os.path.join(artifacts_dir, 'metadata.json'), 'r') as f:
        meta = json.load(f)
        
    with open(os.path.join("outputs", "results.json"), 'r') as f:
        results_json = json.load(f)
        
    MODEL_LOADED = True
except Exception as e:
    print(f"Failed to load artifacts: {e}")
    MODEL_LOADED = False

def predict_probabilities(df_features):
    n = len(df_features)
    test_preds = {'lightgbm': np.zeros((n, 4)), 
                  'extratrees': np.zeros((n, 4)), 
                  'logreg': np.zeros((n, 4))}
    
    for k in models:
        for m in models[k]:
            test_preds[k] += m.predict_proba(df_features)
        test_preds[k] /= len(models[k])
        
    ens_test = (test_preds['lightgbm'] + test_preds['extratrees'] + test_preds['logreg']) / 3
    cal_test = calibrator.predict_proba(np.log(np.clip(ens_test, 1e-15, 1-1e-15)))
    return cal_test

def expected_cost(probs):
    return probs @ COST_MATRIX

class PatientInput(BaseModel):
    gender: Optional[str] = None
    arrival_transport: Optional[str] = None
    temperature: Optional[float] = None
    heartrate: Optional[float] = None
    resprate: Optional[float] = None
    o2sat: Optional[float] = None
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    pain_score: Optional[float] = None
    gcs_eye: Optional[float] = None
    gcs_verbal: Optional[float] = None
    gcs_motor: Optional[float] = None
    avpu: Optional[str] = None
    follows_commands: Optional[float] = None
    chiefcomplaint: Optional[str] = None
    
    # Advanced fields
    bp_unobtainable: Optional[float] = None
    pain_assessable: Optional[float] = None
    consciousness_source: Optional[str] = None
    vs_heartrate_min: Optional[float] = None
    vs_heartrate_max: Optional[float] = None
    vs_heartrate_mean: Optional[float] = None
    vs_resprate_min: Optional[float] = None
    vs_resprate_max: Optional[float] = None
    vs_resprate_mean: Optional[float] = None
    vs_sbp_min: Optional[float] = None
    vs_sbp_max: Optional[float] = None
    vs_sbp_mean: Optional[float] = None
    vs_o2sat_min: Optional[float] = None
    vs_o2sat_max: Optional[float] = None
    vs_o2sat_mean: Optional[float] = None
    vs_temperature_max: Optional[float] = None
    n_vitalsign_readings: Optional[float] = None
    n_diagnoses: Optional[float] = None
    n_home_meds: Optional[float] = None

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_LOADED, "labels": LABELS}

@app.get("/api/schema")
def schema():
    return {
        "core": [
            {"name": "gender", "type": "categorical", "values": ["M", "F"]},
            {"name": "arrival_transport", "type": "categorical", "values": ["AMBULANCE", "WALK IN", "HELICOPTER", "OTHER"]},
            {"name": "temperature", "type": "numeric", "hint": "Celsius", "range": [32.0, 42.0]},
            {"name": "heartrate", "type": "numeric", "hint": "bpm", "range": [0, 200]},
            {"name": "resprate", "type": "numeric", "hint": "bpm", "range": [0, 60]},
            {"name": "o2sat", "type": "numeric", "hint": "%", "range": [0, 100]},
            {"name": "sbp", "type": "numeric", "hint": "mmHg", "range": [0, 250]},
            {"name": "dbp", "type": "numeric", "hint": "mmHg", "range": [0, 150]},
            {"name": "pain_score", "type": "numeric", "hint": "0-10", "range": [0, 10]},
            {"name": "gcs_eye", "type": "numeric", "hint": "1-4", "range": [1, 4]},
            {"name": "gcs_verbal", "type": "numeric", "hint": "1-5", "range": [1, 5]},
            {"name": "gcs_motor", "type": "numeric", "hint": "1-6", "range": [1, 6]},
            {"name": "avpu", "type": "categorical", "values": ["A", "V", "P", "U"]},
            {"name": "follows_commands", "type": "categorical", "values": [0, 1]},
            {"name": "chiefcomplaint", "type": "text"}
        ],
        "advanced": [
            {"name": "bp_unobtainable", "type": "categorical", "values": [0, 1]},
            {"name": "pain_assessable", "type": "categorical", "values": [0, 1]},
            {"name": "consciousness_source", "type": "categorical", "values": ["NURSE", "DOCTOR", "PARAMEDIC", "OTHER"]},
            {"name": "vs_heartrate_min", "type": "numeric"},
            {"name": "vs_heartrate_max", "type": "numeric"},
            {"name": "vs_heartrate_mean", "type": "numeric"},
            {"name": "vs_resprate_min", "type": "numeric"},
            {"name": "vs_resprate_max", "type": "numeric"},
            {"name": "vs_resprate_mean", "type": "numeric"},
            {"name": "vs_sbp_min", "type": "numeric"},
            {"name": "vs_sbp_max", "type": "numeric"},
            {"name": "vs_sbp_mean", "type": "numeric"},
            {"name": "vs_o2sat_min", "type": "numeric"},
            {"name": "vs_o2sat_max", "type": "numeric"},
            {"name": "vs_o2sat_mean", "type": "numeric"},
            {"name": "vs_temperature_max", "type": "numeric"},
            {"name": "n_vitalsign_readings", "type": "numeric"},
            {"name": "n_diagnoses", "type": "numeric"},
            {"name": "n_home_meds", "type": "numeric"}
        ]
    }

@app.post("/api/predict")
def predict(patient: PatientInput):
    if not MODEL_LOADED:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    df = pd.DataFrame([patient.dict(exclude_none=True)])
    
    # Fill in gcs_total if gcs_eye, verbal, motor are present
    if all(k in df.columns for k in ['gcs_eye', 'gcs_verbal', 'gcs_motor']):
        df['gcs_total'] = df['gcs_eye'] + df['gcs_verbal'] + df['gcs_motor']
        
    df_features_raw = make_features(df)
    
    # Reindex to match training feature columns, fill missing with 0.0
    df_features = df_features_raw.reindex(columns=feature_cols, fill_value=0.0)
    
    probs = predict_probabilities(df_features)[0]
    costs = expected_cost(probs)
    
    pred_idx = int(np.argmin(costs))
    argmax_idx = int(np.argmax(probs))
    
    # Determine which START flags fired
    start_flags = []
    if df_features_raw.get('start_resp_high', pd.Series([0]))[0] == 1: start_flags.append('resprate > 30')
    if df_features_raw.get('start_resp_zero', pd.Series([0]))[0] == 1: start_flags.append('resprate == 0')
    if df_features_raw.get('start_hr_zero', pd.Series([0]))[0] == 1: start_flags.append('heartrate == 0')
    if df_features_raw.get('start_sbp_low', pd.Series([0]))[0] == 1: start_flags.append('sbp < 90')
    if df_features_raw.get('start_motor_low', pd.Series([0]))[0] == 1: start_flags.append('gcs_motor < 6')
    
    return {
        "probabilities": {LABELS[i]: float(probs[i]) for i in range(4)},
        "expected_costs": {LABELS[i]: float(costs[i]) for i in range(4)},
        "predicted_label": LABELS[pred_idx],
        "most_likely_label": LABELS[argmax_idx],
        "differs_from_most_likely": pred_idx != argmax_idx,
        "start_flags_fired": start_flags
    }

@app.post("/api/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if file.size and file.size > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 5MB.")
        
    content = await file.read()
    try:
        df = pd.read_csv(io.StringIO(content.decode('utf-8')))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid CSV format")
        
    if 'subject_id' not in df.columns:
        raise HTTPException(status_code=400, detail="CSV must contain 'subject_id' column")
        
    df_features_raw = make_features(df)
    df_features = df_features_raw.reindex(columns=feature_cols, fill_value=0.0)
    
    cal_test = predict_probabilities(df_features)
    costs = expected_cost(cal_test)
    pred_idx = costs.argmin(axis=1)
    
    sub_df = pd.DataFrame({
        'Patient ID': df['subject_id'],
        'Predicted Triage': [LABELS[p] for p in pred_idx],
        'RED Probability': cal_test[:, 0].round(4),
        'YELLOW Probability': cal_test[:, 1].round(4),
        'GREEN Probability': cal_test[:, 2].round(4),
        'BLACK Probability': cal_test[:, 3].round(4)
    })
    
    prob_cols = ['RED Probability', 'YELLOW Probability', 'GREEN Probability', 'BLACK Probability']
    sub_df[prob_cols] = sub_df[prob_cols].div(sub_df[prob_cols].sum(axis=1), axis=0).round(4)
    
    preview = sub_df.head(20).to_dict(orient='records')
    csv_string = sub_df.to_csv(index=False)
    
    # We can't really return a file AND json in the same body natively without multipart or returning JSON with the CSV embedded.
    # To keep it simple, we will return JSON containing the CSV string and the preview.
    return {
        "preview": preview,
        "csv_content": csv_string
    }

@app.get("/api/report")
def report():
    return results_json

@app.get("/api/figures/{name}")
def figures(name: str):
    allowed_figures = ["confusion_matrix.png"]
    if name not in allowed_figures:
        raise HTTPException(status_code=404, detail="Figure not found")
    
    path = os.path.join("outputs", name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Figure not found")
        
    return FileResponse(path)

# Serve frontend in production
if os.path.exists(os.path.join("frontend", "dist")):
    app.mount("/", StaticFiles(directory=os.path.join("frontend", "dist"), html=True), name="frontend")
