import pandas as pd
import numpy as np
import random
import json
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

NUM_SAMPLES = 5000
data = []
services = ["Cardiology", "Surgery", "Pulmonology", "General Medicine"]

for i in range(NUM_SAMPLES):
    service = random.choice(services)

    # ── Vitaux avec distribution RÉALISTE (majorité normale) ──
    # 70% patients stables, 30% critiques
    profile = random.choices(['stable', 'moderate', 'critical'], weights=[0.5, 0.3, 0.2])[0]

    if profile == 'stable':
        temperature    = round(random.uniform(36.0, 37.8), 1)
        heart_rate     = random.randint(60, 95)
        spo2           = random.randint(95, 99)
        bp_systolic    = random.randint(110, 140)
        bp_diastolic   = random.randint(60, 90)
        consciousness  = random.randint(7, 10)   # 0-10, 10=fully alert
        pain_level     = random.randint(0, 4)
        fatigue        = random.randint(0, 2)
        shortness_breath = random.randint(0, 1)

    elif profile == 'moderate':
        temperature    = round(random.uniform(37.5, 39.2), 1)
        heart_rate     = random.randint(90, 120)
        spo2           = random.randint(90, 95)
        bp_systolic    = random.randint(90, 160)
        bp_diastolic   = random.randint(55, 100)
        consciousness  = random.randint(5, 8)
        pain_level     = random.randint(4, 7)
        fatigue        = random.randint(2, 4)
        shortness_breath = random.randint(1, 3)

    else:  # critical
        temperature    = round(random.uniform(38.5, 41.0), 1)
        heart_rate     = random.randint(115, 145)
        spo2           = random.randint(78, 91)
        bp_systolic    = random.randint(75, 95)
        bp_diastolic   = random.randint(40, 60)
        consciousness  = random.randint(0, 5)
        pain_level     = random.randint(7, 10)
        fatigue        = random.randint(3, 5)
        shortness_breath = random.randint(3, 5)

    blood_sugar = round(random.uniform(0.6, 3.0), 2) if random.random() > 0.1 else None
    dressing_changed = random.choice(["Yes", "No"])
    urine_normal     = random.choice(["Yes", "No"])
    medication_taken = random.choice(["Yes", "No"])
    appetite_loss    = random.randint(0, 5)

    # ── Logique de gravité cohérente (consciousness 0-10) ──
    is_critical = (
        spo2 < 88 or
        consciousness <= 3 or
        temperature >= 40.0 or
        pain_level >= 9 or
        heart_rate > 130 or
        bp_systolic < 85
    )
    is_high = (
        spo2 < 92 or
        consciousness <= 5 or
        temperature >= 39.0 or
        heart_rate > 115 or
        pain_level >= 7 or
        shortness_breath >= 4
    )
    is_medium = (
        spo2 < 95 or
        temperature > 38.0 or
        pain_level >= 5 or
        heart_rate > 100 or
        fatigue >= 4 or
        shortness_breath >= 3
    )

    if is_critical:
        gravity_label = "critical"
    elif is_high:
        gravity_label = "high"
    elif is_medium:
        gravity_label = "medium"
    else:
        gravity_label = "low"

    # ── Answers ──
    all_answers = [
        {"question": "What is your pain level?",                    "answer": pain_level},
        {"question": "What is your body temperature (°C)?",         "answer": temperature},
        {"question": "What is your oxygen level (SpO2 %)?",         "answer": spo2},
        {"question": "What is your heart rate (bpm)?",              "answer": heart_rate},
        {"question": "What is your blood pressure (e.g. 120/80)?",  "answer": f"{bp_systolic}/{bp_diastolic}"},
        {"question": "What is your level of consciousness?",        "answer": consciousness},
        {"question": "What is your blood sugar level (mg/dL)?",     "answer": blood_sugar},
        {"question": "Have you changed your dressing?",             "answer": dressing_changed},
        {"question": "Is your urine output normal?",                "answer": urine_normal},
        {"question": "Are you experiencing any shortness of breath?","answer": "Yes" if shortness_breath >= 3 else "No"},
        {"question": "Have you taken all your prescribed medications today?", "answer": medication_taken},
    ]

    # 70% partiels (min 4 questions), 30% complets
    if random.random() < 0.7:
        num_keep = random.randint(4, len(all_answers))
        answers = random.sample(all_answers, num_keep)
    else:
        answers = all_answers

    data.append({
        "sample_id": i + 1,
        "answers": answers,
        "gravity_label": gravity_label,
        "fatigue_score": fatigue,
        "shortness_breath_score": shortness_breath,
    })

df = pd.DataFrame(data)
df.to_json('training_data_large.json', orient='records', indent=2)
df.to_csv('training_data_large.csv', index=False)

print(f"✅ Dataset : {len(df)} exemples")
print(df['gravity_label'].value_counts())