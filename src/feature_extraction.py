# feature_extraction.py
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import os

class FeatureExtractor:
    def __init__(self):
        base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        self.model = base_model

    def extract(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        features = self.model.predict(img_array)
        return features.flatten()

def generate_embeddings(image_folder, save_path="models/embeddings.pkl"):
    import pickle
    fe = FeatureExtractor()
    features = []
    img_paths = []

    for img_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, img_name)
        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            feature = fe.extract(img_path)
            features.append(feature)
            img_paths.append(img_path)

    features = np.array(features)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({"features": features, "paths": img_paths}, f)

    print(f"Embeddings saved to {save_path}")
