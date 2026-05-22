import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import time
from keras.layers import TFSMLayer

st.set_page_config(
    page_title="COVID-19 Detection from Chest X-ray",
    page_icon="🩺",
    layout="centered"
)

st.markdown("""
<style>
.main { background-color: #f5f7fa; }
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
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
model = TFSMLayer("covid_model", call_endpoint="serving_default")

# Categories
categories = ['Covid', 'Normal', 'Viral Pneumonia', 'Not X-ray Image']

# HEADER
st.markdown('<div class="title">Covid-19 Detection Tool</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A CNN based Covid Detection From Chest X-ray Classification System</div>', unsafe_allow_html=True)

# SIDEBAR
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2785/2785819.png", use_column_width=True)
st.sidebar.header("Upload X-ray Image")

uploaded_file = st.sidebar.file_uploader("Choose Chest X-ray Image", type=["jpg", "png", "jpeg"])

# MAIN SECTION
if uploaded_file is not None:
    try:
        # Convert to RGB
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

        # Preprocess
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img = img / 255.0
        img = np.expand_dims(img, axis=0).astype(np.float32)

        # Prediction using TFSMLayer
        with st.spinner("Analyzing X-ray Image..."):
            time.sleep(2)
            outputs = model(img)   # returns a dict
            # Grab the first output tensor
            prediction = list(outputs.values())[0].numpy()

        predicted_class = categories[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Confidence threshold check
        if confidence < 60:
            predicted_class = "Not X-ray Image"

        # Result display
        if predicted_class == "Covid":
            st.error(f"⚠️ Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")
        elif predicted_class == "Normal":
            st.success(f"✅ Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")
        elif predicted_class == "Viral Pneumonia":
            st.warning(f"🫁 Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")
        else:
            st.info(f"❌ Prediction : {predicted_class}\n\nConfidence : {confidence:.2f}%")

        # Probability scores
        st.subheader("Prediction Probabilities")
        for i, category in enumerate(categories):
            st.write(f"{category} : {prediction[0][i]*100:.2f}%")
            st.progress(float(prediction[0][i]))

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("Upload a Chest X-ray Image from Sidebar")
