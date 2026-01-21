from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["concavity_mean"]),
        ]

        scaled_features = scaler.transform([features])
        result = model.predict(scaled_features)

        prediction = "Benign" if result[0] == 1 else "Malignant"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
