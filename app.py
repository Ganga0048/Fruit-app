from flask import Flask, render_template, request
import numpy as np
from PIL import Image
import cv2
import joblib
import os

# Load ML artifacts
model   = joblib.load("model/best_fruit_model.pkl")
scaler  = joblib.load("model/feature_scaler.pkl")
classes = joblib.load("model/fruit_classes.pkl")

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- Feature Extraction (Same as Training) --------
def extract_features_from_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (128, 128))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist([hsv], [0,1,2], None, (8,8,8), [0,180,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# --------------------------- Routes ---------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    uploaded_image_url = None

    if request.method == "POST":
        file = request.files["image"]
        img_pil = Image.open(file).convert("RGB")

        # Save uploaded image to static/uploads
        save_path = os.path.join(UPLOAD_FOLDER, file.filename)
        img_pil.save(save_path)

        # Convert path to URL
        uploaded_image_url = "/" + save_path.replace("\\", "/")

        # Extract features
        feat = extract_features_from_image(img_pil)
        feat_scaled = scaler.transform([feat])

        # Predict
        pred_index = model.predict(feat_scaled)[0]
        prediction = classes[pred_index]

    return render_template("index.html",
                           prediction=prediction,
                           uploaded_image_url=uploaded_image_url)

if __name__ == "__main__":
    app.run(debug=True)
