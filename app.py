"""Streamlit interface for RecycleVision inference."""
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="RecycleVision", page_icon="♻️", layout="centered")

@st.cache_resource
def load_artifacts(model_path: str):
    path = Path(model_path)
    labels = json.loads(path.with_name(path.stem + "_labels.json").read_text())
    return tf.keras.models.load_model(path), labels

st.title("♻️ RecycleVision")
st.caption("Upload a waste image to identify its recycling category.")
model_file = st.sidebar.text_input("Model file", "artifacts/models/mobilenetv2.keras")
uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if not Path(model_file).exists():
        st.warning("Train a model first, or select an existing .keras model in the sidebar.")
    else:
        model, labels = load_artifacts(model_file)
        resized = image.resize((224, 224))
        probabilities = model.predict(np.expand_dims(np.asarray(resized, dtype=np.float32), 0), verbose=0)[0]
        order = probabilities.argsort()[::-1][:3]
        best = order[0]
        st.success(f"Prediction: **{labels[best].title()}** — {probabilities[best]:.1%} confidence")
        st.subheader("Top predictions")
        for idx in order:
            st.progress(float(probabilities[idx]), text=f"{labels[idx].title()}: {probabilities[idx]:.1%}")
else:
    st.info("Upload a photo of a single waste item to begin.")
