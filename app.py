from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load artifacts
model = joblib.load("best_fruit_model.pkl")
scaler = joblib.load("feature_scaler.pkl")
classes = joblib.load("fruit_classes.pkl")

@app.get("/")
def home():
    return {"message": "Fruit model API is alive 🍏"}

@app.post("/predict")
def predict(features: dict):
    # Expecting features = {"data": [1.0, 2.3, ...]}
    x = np.array([features["data"]])
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    label = classes[pred]
    
    return {"prediction": label}
