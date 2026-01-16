from flask import Flask, request, jsonify
import pickle
import cv2
import os
from flask_cors import CORS
from pathlib import Path


from preprocessing.image_pipeline import resize_letterbox, extract_features_mobilenet

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR/"static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CORS(app)


with open(BASE_DIR/"model/naive_bayes_acidentes.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    classes = data["classes"]


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Nenhuma imagem enviada"}), 400

    file = request.files["image"]
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    image = cv2.imread(filepath)
    image = resize_letterbox(image)
    features = extract_features_mobilenet(image)

    probs = model.predict_proba([features])[0]
    pred_index = int(probs.argmax())

    resultado = classes[pred_index]

    probabilidades = {
        classes[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(classes))
    }

    return jsonify({
        "resultado": resultado,
        "probabilidades": probabilidades
    })


if __name__ == "__main__":
    app.run()