import streamlit as st
import numpy as np
import gdown
import os
from PIL import Image
import onnxruntime as ort

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾")
st.title("Classification Chiens vs Chats")

MODEL_PATH = "best_model.onnx"
FILE_ID = "1DHyJ6b7q_cWGNRXl4rlQlmlTvtHb_c_I"  #  ID Google Drive

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return ort.InferenceSession(MODEL_PATH)

session = get_model()

uploaded_file = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((150, 150))
    st.image(img, caption="Image chargée", use_column_width=True)

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_name = session.get_inputs()[0].name
    proba = session.run(None, {input_name: img_array})[0][0][0]

    label = "Dog" if proba > 0.5 else "Cat"
    confiance = proba if proba > 0.5 else 1 - proba
    st.subheader(f"Prédiction : {label}")
    st.write(f"Confiance : {confiance:.2%}")
