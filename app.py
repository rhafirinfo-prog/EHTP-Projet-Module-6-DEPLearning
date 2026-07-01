import streamlit as st
import numpy as np
import gdown
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

st.set_page_config(page_title="Dogs vs Cats Classifier", page_icon="🐾")
st.title("🐶🐱 Classification Chiens vs Chats")

MODEL_PATH = "best_model.keras"
FILE_ID = "1dFFaXCnuAF8ZFiKqiC4k3jBpKNaafJKX"  # ID extrait de ton lien Google Drive

@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"  # ← format correct pour gdown
        gdown.download(url, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = get_model()

uploaded_file = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(150, 150))
    st.image(img, caption="Image chargée", use_column_width=True)
    img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)
    proba = model.predict(img_array)[0][0]
    label = "Dog 🐶" if proba > 0.5 else "Cat 🐱"
    confiance = proba if proba > 0.5 else 1 - proba
    st.subheader(f"Prédiction : {label}")
    st.write(f"Confiance : {confiance:.2%}")