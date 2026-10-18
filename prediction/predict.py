# ============================================================
# STANDALONE PREDICTION MODULE & CLI
# Breast Cancer Histopathology Multi-Class Inference
# ============================================================

import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.config import (
    MODEL_DIR,
    MODEL_CONFIG,
    CLASS_NAMES,
    BENIGN_CLASSES,
    MALIGNANT_CLASSES,
    CLASS_DESCRIPTIONS,
)
from preprocessing.preprocessing import load_and_preprocess_image


def load_model_and_classes(model_key="resnet50"):
    """
    Loads model weights and corresponding class names.
    """
    if model_key not in MODEL_CONFIG:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(MODEL_CONFIG.keys())}")

    config = MODEL_CONFIG[model_key]
    model_path = os.path.join(MODEL_DIR, config["file"])
    class_path = os.path.join(MODEL_DIR, config["class_names_file"])

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)

    class_names = CLASS_NAMES
    if os.path.exists(class_path):
        try:
            with open(class_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list) and len(data) == 8:
                    class_names = [str(x) for x in data]
        except Exception:
            pass

    return model, class_names


def predict(image_path, model_key="resnet50"):
    """
    Runs multi-class histopathology inference on a single image.
    """
    model, class_names = load_model_and_classes(model_key)
    img = load_and_preprocess_image(image_path)
    input_tensor = np.expand_dims(img, axis=0)

    raw_preds = model.predict(input_tensor, verbose=0)[0]

    # Convert logits to probabilities if necessary
    total = float(np.sum(raw_preds))
    if np.min(raw_preds) < 0 or not np.isclose(total, 1.0, atol=0.01):
        raw_preds = tf.nn.softmax(raw_preds).numpy()

    prob_sum = float(np.sum(raw_preds))
    if prob_sum > 0:
        raw_preds = raw_preds / prob_sum

    best_idx = int(np.argmax(raw_preds))
    predicted_class = class_names[best_idx]
    confidence = float(raw_preds[best_idx] * 100)

    tumor_type = "Benign" if predicted_class in BENIGN_CLASSES else "Malignant"
    class_meta = CLASS_DESCRIPTIONS.get(predicted_class, {})

    all_probs = [
        {"class": name, "probability": float(raw_preds[i] * 100)}
        for i, name in enumerate(class_names)
    ]
    all_probs.sort(key=lambda x: x["probability"], reverse=True)

    return {
        "model": MODEL_CONFIG[model_key]["name"],
        "predicted_class": predicted_class,
        "class_index": best_idx,
        "confidence": confidence,
        "tumor_type": tumor_type,
        "description": class_meta.get("description", ""),
        "risk_level": class_meta.get("risk_level", ""),
        "probabilities": all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Histopathology Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to histopathological image")
    parser.add_argument("--model", type=str, default="resnet50", choices=list(MODEL_CONFIG.keys()), help="Model to use")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("BREAST CANCER HISTOPATHOLOGY INFERENCE")
    print("=" * 60)
    print(f"Image: {args.image}")
    print(f"Model: {args.model}")

    result = predict(args.image, args.model)

    print("\nPrediction Result:")
    print("-" * 60)
    print(f"Predicted Class : {result['predicted_class']}")
    print(f"Tumor Category  : {result['tumor_type']}")
    print(f"Confidence      : {result['confidence']:.2f}%")
    print(f"Clinical Action : {result['risk_level']}")
    print("-" * 60)
    print("\nClass Probabilities:")
    for p in result["probabilities"]:
        bar = "=" * int(p["probability"] // 4)
        print(f"  {p['class']:<20}: {p['probability']:6.2f}% [{bar}]")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
