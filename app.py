from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

model_data = joblib.load("best_model.pkl")
model = model_data["model"]
label_encoder = model_data["label_encoder"]

feature_names = ["precipitation", "temp_max", "temp_min", "wind"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = [float(request.form[feature]) for feature in feature_names]

        input_df = pd.DataFrame([input_data], columns=feature_names)

        prediction = model.predict(input_df)[0]

        weather_prediction = label_encoder.inverse_transform([prediction])[0]

        return render_template("index.html", prediction_text=f"สภาพอากาศที่คาดการณ์คือ: {weather_prediction}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"เกิดข้อผิดพลาด: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
