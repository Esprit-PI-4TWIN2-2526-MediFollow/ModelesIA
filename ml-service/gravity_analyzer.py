# gravity-service/gravity_analyzer.py
import httpx
import os
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Use environment variable for production deployment
GRAVITY_SERVICE_URL = os.getenv("GRAVITY_SERVICE_URL", "http://localhost:8001")


class PatientResponseRequest(BaseModel):
    patient_id: str
    patient_name: Optional[str] = None
    answers: List[Dict]


@router.post("/generate")
async def generate_patient_analysis(request: PatientResponseRequest):
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(
            f"{GRAVITY_SERVICE_URL}/predict-gravity",
            json=request.model_dump()
        )
        if resp.status_code != 200:
            return {"status": "error", "detail": resp.text}
        
        data = resp.json()

    features = data.get("features", {})
    gravity = data.get("gravity", "unknown")
    confidence = round(float(data.get("confidence", 0.0)), 1)
    name = data.get("patient_name", request.patient_name or f"Patient {request.patient_id[:8]}")

    return {
        "status": "success",
        "patient_id": request.patient_id,
        "patient_name": name,
        "gravity": gravity,
        "confidence": confidence,
        "analysis": _build_analysis(name, features, gravity, confidence),
        "key_findings": _key_findings(features),
    }


def _build_analysis(name: str, f: dict, gravity: str, confidence: float) -> str:
    text = f"**Clinical Analysis for {name}**\n\n"

    findings = []

    # Protection contre None / NaN
    spo2 = f.get('spo2')
    temp = f.get('temperature')
    pain = f.get('pain_level')
    hr = f.get('heart_rate')
    cons = f.get('consciousness')

    if spo2 is not None and spo2 < 94:
        findings.append(f"Hypoxemia (SpO2 = {spo2}%)")
    if temp is not None and temp > 38.5:
        findings.append(f"Significant Fever ({temp}°C)")
    if pain is not None and pain >= 7:
        findings.append(f"Severe Pain (Level {pain}/10)")
    if hr is not None and hr > 110:
        findings.append(f"Tachycardia ({hr} bpm)")
    if cons is not None and cons < 13:
        findings.append(f"Altered Consciousness (score {cons})")

    if findings:
        text += "**Major Findings:**\n• " + "\n• ".join(findings) + "\n\n"
    else:
        text += "No critical vital signs detected in the provided data.\n\n"

    text += f"**Estimated Gravity Level: {gravity.upper()}** (confidence: {confidence}%)\n\n"
    text += "**Recommendations for the Physician:**\n"

    if gravity == "critical":
        text += "- **Urgent** medical evaluation recommended\n- Continuous vital signs monitoring\n- Consider immediate hospitalization or consultation"
    elif gravity == "high":
        text += "- Close monitoring within 1-2 hours\n- Full re-evaluation recommended"
    elif gravity == "medium":
        text += "- Enhanced surveillance\n- Re-check in 4 hours"
    else:
        text += "- Standard follow-up according to department protocol"

    return text.strip()


def _key_findings(f: dict) -> List[str]:
    findings: List[str] = []

    if f.get('spo2') is not None and f.get('spo2') < 94:
        findings.append("Hypoxemia")

    if f.get('temperature') is not None and f.get('temperature') > 38.5:
        findings.append("Fever")

    if f.get('pain_level') is not None and f.get('pain_level') >= 7:
        findings.append("Severe Pain")

    if f.get('heart_rate') is not None and f.get('heart_rate') > 110:
        findings.append("Tachycardia")

    if f.get('consciousness') is not None and f.get('consciousness') <= 9:
        findings.append("Altered Consciousness")

    if not findings:
        findings.append("No major abnormalities detected")

    return findings