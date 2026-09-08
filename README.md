# AI Soil Health Card - Flask + Tailwind

This UI is inspired by the structure of the supplied Soil Health Card PDF: bilingual header, left-side sample information, nutrient cards, colored status indicators, recommendation table, and a disclaimer strip.

## 1) Save your trained model
In your notebook, after training the best classifier:

```python
import os, joblib
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/crop_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")
```

Copy the two `.pkl` files into this project's `models/` folder.

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Optional Gemini setup
Copy `.env.example` to `.env` and add your key:

```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Never hardcode the API key in HTML or commit it to GitHub.

## 4) Run

```bash
python app.py
```

Open: http://127.0.0.1:5000

## Expected feature order
The Flask app sends features in this order:

`N, P, K, temperature, humidity, ph, rainfall`

This must match the order used when your model was trained.
