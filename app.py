import json
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import joblib
import numpy as np
import pandas as pd

load_dotenv()

app = Flask(__name__)

# --- Paths & Config ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "models", "crop_model.pkl"))
ENCODER_PATH = os.getenv("ENCODER_PATH", os.path.join(BASE_DIR, "models", "label_encoder.pkl"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH_Value", "Rainfall"]
THRESHOLDS = {
    "N": (40, 120), "P": (20, 80), "K": (20, 100),
    "temperature": (18, 35), "humidity": (40, 85),
    "ph": (5.5, 7.5), "rainfall": (50, 250)
}

# --- Load Model Artifacts ---
model, label_encoder, model_error = None, None, None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)
except Exception as exc:
    model_error = str(exc)


# --- Core Helpers ---
def decode_classes(class_values):
    arr = np.asarray(class_values)
    if label_encoder is None:
        return [str(x) for x in arr]
    try:
        return list(label_encoder.inverse_transform(arr.astype(int)))
    except Exception:
        return [str(x) for x in arr]


def top_predictions(input_df, top_k=3):
    if model is None:
        raise RuntimeError("Model is not loaded. Place crop_model.pkl in models/")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(input_df)[0]
        indices = np.argsort(probs)[::-1][:top_k]
        classes = getattr(model, "classes_", np.arange(len(probs)))
        crop_names = decode_classes([classes[i] for i in indices])

        return [
            {"crop": crop_names[i], "confidence": round(float(probs[idx]) * 100, 2)}
            for i, idx in enumerate(indices)
        ]

    predicted = model.predict(input_df)
    return [{"crop": decode_classes(predicted)[0], "confidence": None}]


def classify_status(val, low, high):
    if val < low:
        return "low"
    if val > high:
        return "high"
    return "good"


def get_gemini_response(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return res.text.strip() if res.text else None
    except Exception as exc:
        return f"Gemini error: {exc}"


# --- Routes ---
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", model_error=model_error)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = {
            "N": float(request.form["N"]),
            "P": float(request.form["P"]),
            "K": float(request.form["K"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "ph": float(request.form["ph"]),
            "rainfall": float(request.form["rainfall"]),
        }

        input_df = pd.DataFrame([{
            "Nitrogen": values["N"],
            "Phosphorus": values["P"],
            "Potassium": values["K"],
            "Temperature": values["temperature"],
            "Humidity": values["humidity"],
            "pH_Value": values["ph"],
            "Rainfall": values["rainfall"]
        }])[FEATURES]

        predictions = top_predictions(input_df, top_k=3)

        prompt = f"""You are an agricultural decision-support assistant.
Analyze these inputs and recommendations under 250 words:
INPUTS: {json.dumps(values)}
PREDICTIONS: {json.dumps(predictions)}

Format using exactly these four section titles:
Soil Health Summary
Why This Crop Fits
Practical Next Steps
Caution
Rules: Plain text only, practical advice, no guaranteed yields, no dangerous chemical rates."""

        advice = get_gemini_response(prompt)
        statuses = {k: classify_status(values[k], *bounds) for k, bounds in THRESHOLDS.items()}

        return render_template(
            "index.html",
            values=values,
            predictions=predictions,
            advice=advice,
            statuses=statuses,
            error=None,
            model_error=model_error
        )

    except Exception as exc:
        return render_template(
            "index.html",
            values=None,
            predictions=None,
            advice=None,
            statuses=None,
            error=f"Prediction error: {str(exc)}",
            model_error=model_error
        ), 400


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"success": False, "error": "Please enter a question."}), 400

        soil_values = data.get("soil_values", {})
        predictions = data.get("predictions", [])

        prompt = f"""You are an AI agricultural assistant.
DATA: {json.dumps(soil_values)}
RECOMMENDATIONS: {json.dumps(predictions)}
QUESTION: {message}

Answer directly in simple language. Explain agronomic reasoning using the data provided. Keep concise."""

        answer = get_gemini_response(prompt)
        if not answer:
            return jsonify({"success": False, "error": "Gemini API unavailable or unconfigured."}), 500

        return jsonify({"success": True, "answer": answer})

    except Exception:
        return jsonify({"success": False, "error": "Unable to generate a response right now."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
