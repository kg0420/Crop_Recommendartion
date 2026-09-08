import os
import json
import joblib
import numpy as np
import pandas as pd

from flask import Flask, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(BASE_DIR, "models", "crop_model.pkl")
)

ENCODER_PATH = os.getenv(
    "ENCODER_PATH",
    os.path.join(BASE_DIR, "models", "label_encoder.pkl")
)

# ============================================================
# IMPORTANT:
# These MUST match the columns used while training your model.
# Your notebook uses exactly these columns.
# ============================================================

FEATURES = [
    "Nitrogen",
    "Phosphorus",
    "Potassium",
    "Temperature",
    "Humidity",
    "pH_Value",
    "Rainfall"
]

# ============================================================
# LOAD MODEL
# ============================================================

model = None
label_encoder = None
model_error = None

try:

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if os.path.exists(ENCODER_PATH):
        label_encoder = joblib.load(ENCODER_PATH)

except Exception as exc:

    model_error = str(exc)


# ============================================================
# DECODE PREDICTIONS
# ============================================================

def decode_classes(class_values):

    arr = np.asarray(class_values)

    if label_encoder is None:
        return [str(x) for x in arr]

    try:

        return list(
            label_encoder.inverse_transform(
                arr.astype(int)
            )
        )

    except Exception:

        return [str(x) for x in arr]


# ============================================================
# TOP PREDICTIONS
# ============================================================

def top_predictions(input_df, top_k=3):

    if model is None:

        raise RuntimeError(
            "Model was not loaded. "
            "Please put crop_model.pkl inside models/"
        )

    # Models such as RandomForest, Naive Bayes,
    # XGBoost, etc. can provide probabilities.

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(input_df)[0]

        # Highest probability first
        indices = np.argsort(probabilities)[::-1][:top_k]

        classes = getattr(
            model,
            "classes_",
            np.arange(len(probabilities))
        )

        selected_classes = [
            classes[i]
            for i in indices
        ]

        crop_names = decode_classes(selected_classes)

        predictions = []

        for i, index in enumerate(indices):

            predictions.append({
                "crop": crop_names[i],
                "confidence": round(
                    float(probabilities[index]) * 100,
                    2
                )
            })

        return predictions

    # Fallback for models without predict_proba

    predicted = model.predict(input_df)

    crop = decode_classes(predicted)[0]

    return [
        {
            "crop": crop,
            "confidence": None
        }
    ]


# ============================================================
# GEMINI
# ============================================================

def generate_gemini_advice(values, predictions):

    api_key = os.getenv("GEMINI_API_KEY")

    # Gemini is optional
    if not api_key:
        return None

    try:

        from google import genai

        client = genai.Client(
            api_key=api_key
        )

        model_name = os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash"
        )

        prompt = f"""
You are an agricultural decision-support assistant.

A machine-learning crop recommendation model analyzed
the following soil and weather measurements.

SOIL AND WEATHER INPUT:

{json.dumps(values, indent=2)}

MODEL PREDICTIONS:

{json.dumps(predictions, indent=2)}

Generate a concise farmer-friendly agricultural analysis.

Use exactly these four section titles:

Soil Health Summary
Why This Crop Fits
Practical Next Steps
Caution

Rules:

- The ML prediction is a recommendation, not certainty.
- Do not invent measurements.
- Do not claim that a crop is guaranteed to produce a good yield.
- Explain the relationship between the supplied soil/weather values
  and the recommended crop.
- Consider the top prediction and alternatives.
- Give practical but general agricultural guidance.
- Mention that local agronomist/soil-lab advice and current weather
  should also be considered.
- Do not provide dangerous or highly specific chemical application
  instructions.
- Keep the response below 250 words.
- Plain text only.
"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )

        text = response.text

        if text:
            return text.strip()

        return None

    except Exception as exc:

        return (
            "Gemini advice could not be generated.\n\n"
            f"Reason: {exc}"
        )


# ============================================================
# VISUAL STATUS
# ============================================================

def classify_status(value, low=None, high=None):

    if low is not None and value < low:
        return "low"

    if high is not None and value > high:
        return "high"

    return "good"


# ============================================================
# HOME
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return render_template(
        "index.html",
        values=None,
        predictions=None,
        advice=None,
        statuses=None,
        error=None,
        model_error=model_error
    )


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        # ====================================================
        # Read values from HTML form
        # ====================================================

        values = {
            "N": float(request.form["N"]),
            "P": float(request.form["P"]),
            "K": float(request.form["K"]),
            "temperature": float(request.form["temperature"]),
            "humidity": float(request.form["humidity"]),
            "ph": float(request.form["ph"]),
            "rainfall": float(request.form["rainfall"])
        }

        # ====================================================
        # IMPORTANT:
        # Column names MUST exactly match training dataset
        # ====================================================

        input_df = pd.DataFrame([{
            "Nitrogen": values["N"],
            "Phosphorus": values["P"],
            "Potassium": values["K"],
            "Temperature": values["temperature"],
            "Humidity": values["humidity"],
            "pH_Value": values["ph"],
            "Rainfall": values["rainfall"]
        }])

        print("\n==============================")
        print("INPUT DATA")
        print("==============================")
        print(input_df)

        print("\nFEATURE NAMES:")
        print(input_df.columns.tolist())

        print("\nMODEL EXPECTS:")
        if hasattr(model, "feature_names_in_"):
            print(model.feature_names_in_.tolist())

        # ====================================================
        # Prediction
        # ====================================================

        predictions = top_predictions(
            input_df,
            top_k=3
        )

        print("\n==============================")
        print("PREDICTIONS")
        print("==============================")
        print(predictions)

        # ====================================================
        # Gemini
        # ====================================================

        advice = generate_gemini_advice(
            values,
            predictions
        )

        # ====================================================
        # UI STATUS
        # ====================================================

        statuses = {

            "N": classify_status(
                values["N"],
                40,
                120
            ),

            "P": classify_status(
                values["P"],
                20,
                80
            ),

            "K": classify_status(
                values["K"],
                20,
                100
            ),

            "temperature": classify_status(
                values["temperature"],
                18,
                35
            ),

            "humidity": classify_status(
                values["humidity"],
                40,
                85
            ),

            "ph": classify_status(
                values["ph"],
                5.5,
                7.5
            ),

            "rainfall": classify_status(
                values["rainfall"],
                50,
                250
            )
        }

        # ====================================================
        # Render result
        # ====================================================

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

        print("\n==============================")
        print("PREDICTION ERROR")
        print("==============================")
        print(exc)

        return render_template(
            "index.html",
            values=None,
            predictions=None,
            advice=None,
            statuses=None,
            error=f"Prediction error: {str(exc)}",
            model_error=model_error
        ), 400

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )