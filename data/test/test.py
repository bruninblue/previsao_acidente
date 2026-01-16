from pathlib import Path
import pickle
import cv2
import numpy as np
from skimage.feature import hog
import os

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR /"model"


with open(MODEL_PATH/"naive_bayes_acidentes.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
classes = data["classes"]

print(classes)
print(model)

# iniciando teste


def resize_letterbox(image, target_size=(128, 128), color=(114, 114, 114)):
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return padded


# extração de Features (HOG)
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features




def predict_image(img_path):
    image = cv2.imread(str(img_path))

    if image is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {img_path}")

    image = resize_letterbox(image)
    features = extract_hog(image)

    pred = model.predict([features])[0]
    return classes[pred]

IMAGE_PATH = BASE_DIR/"image_processing"/"dataset"/"test"/"moderado"

resultado = predict_image(IMAGE_PATH/"frame_012225_PNG_jpg.rf.9d02739e8dd4862f5c63320a2cabdb4d.jpg")
print("Predição:", resultado)
