from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(request.form.get(f)) for f in [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time'
        ]]
        prediction = model.predict([features])[0]
        result = "The patient is at risk of death." if prediction == 1 else "The patient is likely to survive."
        return render_template("index.html", prediction_text=result)
    except Exception as e:
        return render_template("index.html", prediction_text="Error: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
