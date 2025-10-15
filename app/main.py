# main.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pickle
from PIL import Image
from src.feature_extraction import FeatureExtractor
from src.similarity_search import get_top_similar

st.set_page_config(page_title="Similarity check", layout="wide")
st.title("Visual Product Matcher")
st.write("Upload a product image to find visually similar products.")

fe = FeatureExtractor()

try:
    with open("models/embeddings.pkl", "rb") as f:
        data = pickle.load(f)
    features = data.get("features", [])
    paths = data.get("paths", [])
except FileNotFoundError:
    st.error("Embeddings file not found. Please generate embeddings first!")
    st.stop()

if len(features) == 0:
    st.warning("No images found in the dataset!")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.subheader("Uploaded Image")
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Your Image", width=250)
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract features for uploaded image
    query_feature = fe.extract(temp_path)
    results = get_top_similar(query_feature, features, paths, top_k=5)

    if len(results) == 0:
        st.info("No matching products found.")
    else:
        fixed_size = (200, 200)
        st.subheader("Similar Products Available:")
        cols = st.columns(len(results))
        for idx, (path, score) in enumerate(results):
            with cols[idx]:
                img = Image.open(path).convert('RGB')
                img = img.resize(fixed_size)
                st.image(img, use_container_width=True)
                st.caption(f"Similarity Index: {score:.4f}")
