"""
Run these lines in the SAME notebook/session after training your best classifier.
Replace best_model with your trained model variable name.
"""
import os
import joblib

os.makedirs("models", exist_ok=True)

# Example:
# best_model = models["XGBoost"]
# best_model.fit(x_train, y_train)

joblib.dump(best_model, "models/crop_model.pkl")
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Saved crop_model.pkl and label_encoder.pkl")
