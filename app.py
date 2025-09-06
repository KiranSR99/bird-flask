import os
import numpy as np
import pickle
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS
from tensorflow.keras.applications.mobilenet import preprocess_input
from PIL import Image

# Import your scratch CNN classes
from train_scratch import SimpleCNN, Conv2D, Dense, Flatten, MaxPool2D, ReLU

# ----------------------------
# Config
# ----------------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Load Models
# ----------------------------
bird_model = load_model("models/bird_model.keras")         # Custom Keras CNN
mobilenet_model = load_model("models/mobilenet_model.h5") # MobileNet
with open("models/scratch_model.pkl", "rb") as f:         # Scratch CNN (NumPy)
    scratch_model = pickle.load(f)

# ----------------------------
# Load Class Names
# ----------------------------
def load_class_names(file_path):
    class_names = []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                class_names.append(parts[1])
    return class_names

BIRD_CLASS_NAMES = load_class_names("bird/lists/classes.txt")
MOBILENET_CLASS_NAMES = load_class_names("bird/lists/classes.txt")
SCRATCH_CLASS_NAMES = load_class_names("bird/lists/classes.txt")

# ----------------------------
# Helpers
# ----------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_bird_model(img_path):
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_mobilenet(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def preprocess_scratch_model(img_path):
    img = Image.open(img_path).convert("RGB").resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)
    img_array = np.expand_dims(img_array, axis=0)   # (1, C, H, W)
    return img_array

def clean_bird_name(raw_name: str) -> str:
    if "." in raw_name:
        raw_name = raw_name.split(".", 1)[1]
    return raw_name.replace("_", " ")

def get_top_predictions(preds, class_names, top=3):
    probs = preds.flatten()
    top_indices = probs.argsort()[-top:][::-1]
    return [
        {"species": clean_bird_name(class_names[i]), "probability": float(probs[i])}
        for i in top_indices
    ]

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ----------------------------
# Routes
# ----------------------------
@app.route("/predict/<model_name>", methods=["POST"])
def predict(model_name):
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    file.save(filepath)

    try:
        if model_name == "bird":
            img_array = preprocess_bird_model(filepath)
            preds = bird_model.predict(img_array)
            top_preds = get_top_predictions(preds, BIRD_CLASS_NAMES)

        elif model_name == "mobilenet":
            img_array = preprocess_mobilenet(filepath)
            preds = mobilenet_model.predict(img_array)
            top_preds = get_top_predictions(preds, MOBILENET_CLASS_NAMES)

        elif model_name == "scratch":
            img_array = preprocess_scratch_model(filepath)
            logits = scratch_model.forward(img_array)  # use forward
            probs = softmax(logits)                    # convert to probabilities
            top_preds = get_top_predictions(probs, SCRATCH_CLASS_NAMES)

        else:
            return jsonify({"error": "Unknown model"}), 400

        return jsonify({"model": model_name, "predictions": top_preds})

    except Exception as e:
        print("Prediction Error:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# ----------------------------
# Run Server
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
