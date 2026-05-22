import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time

st.set_page_config(
    page_title="COVID-19 Detection from Chest X-ray",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    background-color: #0033cc;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
    color: yellow;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: gray;
    margin-bottom: 30px;
}
.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
model = tf.keras.models.load_model("covid_xray_model.h5")
categories = ['Covid', 'Normal', 'Viral Pneumonia']

# HEADER
st.markdown('<div class="title">Covid-19 Detection App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A CNN based Covid Detection From Chest X-ray Classification System</div>', unsafe_allow_html=True)

# SIDEBAR
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", use_column_width=True)
st.sidebar.header("Upload X-ray Image")

uploaded_file = st.sidebar.file_uploader("Choose Chest X-ray Image", type=["jpg", "png", "jpeg"])

# MAIN SECTION
if uploaded_file is not None:
    try:
        # Always convert to RGB (fixes grayscale and RGBA issues)
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

        # Preprocess
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0) 

        # Debug: show shape
        st.write(f"Image shape after preprocessing: {img.shape}")

        # Prediction
        with st.spinner("Analyzing X-ray Image..."):
            time.sleep(2)
            prediction = model.predict(img)

        predicted_class = categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Result display
        if predicted_class == "Covid":
            st.error(f"⚠️ Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")
        elif predicted_class == "Normal":
            st.success(f"✅ Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")
        else:
            st.warning(f"🫁 Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")

        # Probability scores
        st.subheader("Prediction Probabilities")
        for i, category in enumerate(categories):
            st.write(f"{category} : {prediction[0][i]*100:.2f}%")
            st.progress(float(prediction[0][i]))

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("Upload a Chest X-ray Image from Sidebar")
