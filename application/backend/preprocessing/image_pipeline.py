import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input  # type: ignore

# Inicializa o modelo da Google uma única vez para economizar memória
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))

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

def extract_features_mobilenet(image):
    # Converte BGR (OpenCV) para RGB (MobileNetV2)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img_rgb, axis=0)
    # Pré-processamento obrigatório da arquitetura CNN 
    img_array = preprocess_input(img_array.astype(np.float32))
    # Extrai o vetor de 1280 características numéricas
    features = base_model.predict(img_array, verbose=0)
    return features.flatten()