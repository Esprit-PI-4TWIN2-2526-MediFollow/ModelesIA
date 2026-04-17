# gravity-service/main.py

import joblib
import json
import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional

app = FastAPI(title="Gravity ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Charger le modèle ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = joblib.load(os.path.join(BASE_DIR, "models/gravity_model.pkl"))

with open(os.path.join(BASE_DIR, "models/feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

print("✅ Gravity model loaded successfully")

# ── CHARGEMENT DES MÉDIANES (ajoute après le chargement du modèle) ──
with open(os.path.join(BASE_DIR, "models/median_imputer.json")) as f:
    MEDIANS = json.load(f)
print("✅ Median imputer loaded successfully")
# ── Schéma d'entrée ──────────────────────────────────────────────
class GravityRequest(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    answers: List[Dict]

# ── NOUVELLE VERSION de extract_features (identique à celle du training) ──
def extract_features(answers: list) -> dict:
    features = {
        'pain_level': np.nan,
        'temperature': np.nan,
        'spo2': np.nan,
        'heart_rate': np.nan,
        'bp_systolic': np.nan,
        'bp_diastolic': np.nan,
        'consciousness': np.nan,
        'blood_sugar': np.nan,
        'dressing_changed': 0,
        'urine_normal': 1,
        'fatigue_score': 0,
        'shortness_breath_score': 0,
    }

    for ans in answers:
        q = str(ans.get('question', '')).lower()
        val = ans.get('answer')
        try:
            if 'pain level' in q:
                features['pain_level'] = float(val)
            elif 'temperature' in q:
                features['temperature'] = float(val)
            elif any(x in q for x in ['oxygen', 'spo2']):
                features['spo2'] = float(val)
            elif 'heart rate' in q:
                features['heart_rate'] = float(val)
            elif 'blood pressure' in q:
                if isinstance(val, str) and '/' in val:
                    s, d = val.split('/')
                    features['bp_systolic'] = float(s.strip())
                    features['bp_diastolic'] = float(d.strip())
            elif 'consciousness' in q:
                features['consciousness'] = float(val)
            elif any(x in q for x in ['sugar', 'glycémie']):
                if isinstance(val, (int, float)):
                    features['blood_sugar'] = float(val)
            elif 'dressing' in q:
                features['dressing_changed'] = 1 if str(val).lower() == 'yes' else 0
            elif 'urine' in q:
                features['urine_normal'] = 1 if str(val).lower() == 'yes' else 0
        except:
            continue
 # ── Mêmes seuils que le training ──
    spo2 = features['spo2']
    hr   = features['heart_rate']
    sbp  = features['bp_systolic']
    temp = features['temperature']
    pain = features['pain_level']
    # Features dérivées (identique au training → gère NaN correctement)
    features['hypoxemia']   = 0 if np.isnan(spo2) else int(spo2 < 94)
    features['tachycardia'] = 0 if np.isnan(hr)   else int(hr > 100)
    features['hypotension'] = 0 if np.isnan(sbp)  else int(sbp < 90)
    features['fever']       = 0 if np.isnan(temp) else int(temp > 38.0)
    features['severe_pain'] = 0 if np.isnan(pain) else int(pain >= 7)

    return features

# ── Endpoint principal ───────────────────────────────────────────
@app.post("/predict-gravity")
def predict_gravity(req: GravityRequest):
    features = extract_features(req.answers)          # ← raw (peut contenir NaN)

    df = pd.DataFrame([features])
    df = df[FEATURE_NAMES]
    
    # ← NOUVELLE IMPUTATION (la clé du fix)
    df = df.fillna(MEDIANS)

    gravity    = model.predict(df)[0]
    probas     = model.predict_proba(df)[0]
    confidence = float(max(probas) * 100)

    display_name = req.patient_name or f"Patient {req.patient_id[:8]}"

    print(f"🔍 Patient: {display_name} → gravity={gravity}, confidence={round(confidence,1)}%")

    return {
        "status": "success",
        "patient_id": req.patient_id,
        "patient_name": display_name,
        "gravity": gravity,
        "confidence": round(confidence, 1),
        "features": {
            k: (None if isinstance(v, float) and np.isnan(v) else v)
            for k, v in features.items()   # on renvoie les vraies valeurs (NaN→None)
        }
    }

@app.get("/health")
def health():
    import sklearn
    return {
        "status":   "ok",
        "model":    "gravity_model.pkl",
        "sklearn":  sklearn.__version__
    }