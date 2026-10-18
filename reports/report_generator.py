# ============================================================
# CLINICAL DIAGNOSTIC REPORT GENERATOR
# Breast Cancer XAI Framework (BreakHis 400X)
# ============================================================

import os
import sys
import json
import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.config import BENIGN_CLASSES, MALIGNANT_CLASSES, CLASS_DESCRIPTIONS, REPORTS_DIR


def generate_clinical_report(prediction_record, patient_info=None):
    """
    Generates a structured clinical pathology report text and metadata
    from a model prediction record.
    """
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pred_class = prediction_record.get("predicted_class", "unknown")
    confidence = float(prediction_record.get("confidence", 0.0))
    model_name = prediction_record.get("model_name", "ResNet50")
    image_name = prediction_record.get("image_name", "biopsy_sample.png")
    gradcam_name = prediction_record.get("gradcam_name")

    is_malignant = pred_class in MALIGNANT_CLASSES
    category = "MALIGNANT" if is_malignant else "BENIGN"

    meta = CLASS_DESCRIPTIONS.get(pred_class, {
        "description": "Histopathological subtype characterized by specific cellular patterns.",
        "risk_level": "Consult pathologist for confirmation.",
    })

    patient_name = (patient_info or {}).get("name", "De-Identified Patient")
    biopsy_id = (patient_info or {}).get("biopsy_id", f"BX-{prediction_record.get('id', '001')}")

    report_text = f"""
================================================================================
                    BREAST CANCER HISTOPATHOLOGY REPORT
================================================================================
Report Date: {now}
Institution: Advanced Histopathology & XAI Diagnostic Laboratory
Biopsy Sample ID: {biopsy_id}
Patient Reference: {patient_name}
Input Specimen: {image_name} (BreakHis 400X Optical Magnification)
Analysis Model: {model_name} Deep Convolutional Neural Network
--------------------------------------------------------------------------------
1. PRIMARY DIAGNOSTIC FINDINGS
--------------------------------------------------------------------------------
Classified Subtype : {pred_class.upper()}
Diagnostic Status  : {category}
Model Confidence   : {confidence:.2f}%
XAI Localization   : {"Grad-CAM Heatmap Available" if gradcam_name else "Standard Inference"}

--------------------------------------------------------------------------------
2. PATHOLOGICAL & ARCHITECTURAL DESCRIPTION
--------------------------------------------------------------------------------
{meta['description']}

--------------------------------------------------------------------------------
3. RISK STRATIFICATION & CLINICAL GUIDANCE
--------------------------------------------------------------------------------
Clinical Risk: {meta['risk_level']}

Recommendation:
- This automated assessment is an auxiliary diagnostic decision support tool.
- Confirmation via microscopic examination and immunohistochemistry (ER, PR, HER2)
  by a certified pathologist is mandatory prior to treatment initiation.

================================================================================
                  CONFIDENTIAL - FOR CLINICAL RESEARCH USE ONLY
================================================================================
"""

    return {
        "generated_at": now,
        "biopsy_id": biopsy_id,
        "patient_name": patient_name,
        "predicted_class": pred_class,
        "category": category,
        "confidence": confidence,
        "model_name": model_name,
        "image_name": image_name,
        "gradcam_name": gradcam_name,
        "description": meta["description"],
        "risk_level": meta["risk_level"],
        "report_text": report_text.strip(),
    }


def save_report_to_file(report_data, output_filename=None):
    """
    Saves the generated report as a clinical text file in the reports directory.
    """
    if not output_filename:
        output_filename = f"report_{report_data['biopsy_id']}.txt"

    os.makedirs(REPORTS_DIR, exist_ok=True)
    file_path = os.path.join(REPORTS_DIR, output_filename)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_data["report_text"])

    return file_path


if __name__ == "__main__":
    sample_pred = {
        "id": 101,
        "predicted_class": "ductal_carcinoma",
        "confidence": 97.45,
        "model_name": "ResNet50",
        "image_name": "SOB_M_DC-14-22549AB-400-001.png",
        "gradcam_name": "gradcam_sample.jpg"
    }
    rep = generate_clinical_report(sample_pred)
    print(rep["report_text"])
