from typing import Optional
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
#from patient_analyzer import router as analysis_router
from gravity_analyzer import router as gravity_router
app = FastAPI(title="MediFollow ML Service")

app.include_router(gravity_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Charger les modèles de l'alerte──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, 'medifollow_scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'medifollow_pca.pkl'), 'rb') as f:
    pca = pickle.load(f)

with open(os.path.join(BASE_DIR, 'medifollow_model.pkl'), 'rb') as f:
    model = pickle.load(f)

print("✅ Models loaded successfully")

# ── Schéma d'entrée — tous les champs sont optionnels ────────
class PatientVitals(BaseModel):
    heartRate:   Optional[float] = None
    spo2:        Optional[float] = None
    temperature: Optional[float] = None
    systolicBP:  Optional[float] = None
    diastolicBP: Optional[float] = None

# ── Endpoint principal ───────────────────────────────────────
@app.post("/predict-alert")
def predict_alert(vitals: PatientVitals):

    # Vérifier qu'on a au moins les données essentielles
    missing = [
        k for k, v in {
            'heartRate':   vitals.heartRate,
            'spo2':        vitals.spo2,
            'temperature': vitals.temperature,
            'systolicBP':  vitals.systolicBP,
            'diastolicBP': vitals.diastolicBP,
        }.items() if v is None
    ]

    # Si trop de données manquantes → pas d'alerte
    if len(missing) >= 4:
        return {
            "hasAlert":          False,
            "severity":          "low",
            "alertProbability":  0.0,
            "normalProbability": 100.0,
            "reason":            f"Insufficient data: {missing}"
        }

    # Valeurs par défaut pour les champs manquants
    heartRate   = vitals.heartRate   if vitals.heartRate   is not None else 80.0
    spo2        = vitals.spo2        if vitals.spo2        is not None else 98.0
    temperature = vitals.temperature if vitals.temperature is not None else 37.0
    systolicBP  = vitals.systolicBP  if vitals.systolicBP  is not None else 120.0
    diastolicBP = vitals.diastolicBP if vitals.diastolicBP is not None else 80.0

    # Calcul MAP (Mean Arterial Pressure)
    map_val = (systolicBP + 2 * diastolicBP) / 3

    # Features dans le même ordre que l'entraînement
    X = pd.DataFrame([{
        'Heart Rate (bpm)':                    heartRate,
        'SpO2 Level (%)':                      spo2,
        'Body Temperature (°C)':               temperature,
        'Systolic Blood Pressure (mmHg)':      systolicBP,
        'Diastolic Blood Pressure (mmHg)':     diastolicBP,
        'MAP':                                 map_val
    }])

    # Pipeline : Scale → PCA → Predict
    X_scaled   = scaler.transform(X)
    X_pca      = pca.transform(X_scaled)
    prediction = model.predict(X_pca)[0]
    proba      = model.predict_proba(X_pca)[0]

    alert_proba = float(proba[1])

    # Sévérité selon la probabilité
    if alert_proba >= 0.85:
        severity = 'critical'
    elif alert_proba >= 0.65:
        severity = 'high'
    elif alert_proba >= 0.45:
        severity = 'medium'
    else:
        severity = 'low'

    print(f"🔍 Vitals reçus: HR={heartRate}, SpO2={spo2}, Temp={temperature}, "
          f"SBP={systolicBP}, DBP={diastolicBP}")
    print(f"🚨 Résultat: hasAlert={bool(prediction==1)}, "
          f"severity={severity}, probability={round(alert_proba*100,1)}%")

    return {
        "hasAlert":          bool(prediction == 1),
        "severity":          severity,
        "alertProbability":  round(alert_proba * 100, 1),
        "normalProbability": round(float(proba[0]) * 100, 1),
    }

@app.get("/health")
def health():
    return {"status": "ok", "models": "loaded"}