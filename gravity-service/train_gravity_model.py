import pandas as pd
import numpy as np
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

print("Chargement du dataset...")
df = pd.read_json('training_data_large.json')
print(f"Dataset : {len(df)} exemples")
print(df['gravity_label'].value_counts())

def extract_features(row):
    answers = row['answers']
    features = {
        'pain_level':             np.nan,
        'temperature':            np.nan,
        'spo2':                   np.nan,
        'heart_rate':             np.nan,
        'bp_systolic':            np.nan,
        'bp_diastolic':           np.nan,
        'consciousness':          np.nan,
        'blood_sugar':            np.nan,
        'dressing_changed':       0,
        'urine_normal':           1,
        'fatigue_score':          row.get('fatigue_score', 0),
        'shortness_breath_score': row.get('shortness_breath_score', 0),
    }

    for ans in answers:
        q   = str(ans.get('question', '')).lower()
        val = ans.get('answer')
        try:
            if 'pain level'    in q: features['pain_level']    = float(val)
            elif 'temperature' in q: features['temperature']   = float(val)
            elif any(x in q for x in ['oxygen', 'spo2']):
                                     features['spo2']          = float(val)
            elif 'heart rate'  in q: features['heart_rate']    = float(val)
            elif 'blood pressure' in q:
                if isinstance(val, str) and '/' in val:
                    s, d = val.split('/')
                    features['bp_systolic']  = float(s.strip())
                    features['bp_diastolic'] = float(d.strip())
            elif 'consciousness' in q: features['consciousness'] = float(val)
            elif any(x in q for x in ['sugar', 'blood sugar']):
                if isinstance(val, (int, float)): features['blood_sugar'] = float(val)
            elif 'dressing' in q:
                features['dressing_changed'] = 1 if str(val).lower() == 'yes' else 0
            elif 'urine' in q:
                features['urine_normal'] = 1 if str(val).lower() == 'yes' else 0
        except: continue

    # ── Features dérivées — seuils UNIQUES pour train ET prod ──
    spo2 = features['spo2']
    hr   = features['heart_rate']
    sbp  = features['bp_systolic']
    temp = features['temperature']
    pain = features['pain_level']

    features['hypoxemia']   = 0 if np.isnan(spo2) else int(spo2 < 94)
    features['tachycardia'] = 0 if np.isnan(hr)   else int(hr > 100)
    features['hypotension'] = 0 if np.isnan(sbp)  else int(sbp < 90)
    features['fever']       = 0 if np.isnan(temp) else int(temp > 38.0)
    features['severe_pain'] = 0 if np.isnan(pain) else int(pain >= 7)

    return pd.Series(features)

print("Extraction des features...")
feature_df = df.apply(extract_features, axis=1)

# Médianes sur les vraies données (avant imputation)
medians = feature_df.median().to_dict()
print("Médianes du dataset :")
for k, v in medians.items():
    print(f"  {k}: {v}")

feature_df = feature_df.fillna(feature_df.median())

X = feature_df
y = df['gravity_label']

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print("Distribution après SMOTE :", pd.Series(y_res).value_counts().to_dict())

X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

model = RandomForestClassifier(
    n_estimators=400, max_depth=20,
    min_samples_split=5, class_weight='balanced',
    random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nÉvaluation :")
print(classification_report(y_test, model.predict(X_test)))
print(confusion_matrix(y_test, model.predict(X_test)))

joblib.dump(model, 'models/gravity_model.pkl')
with open('models/feature_names.json', 'w') as f:
    json.dump(list(X.columns), f)
with open('models/median_imputer.json', 'w') as f:
    json.dump(medians, f)

print("✅ Modèle + médianes sauvegardés")