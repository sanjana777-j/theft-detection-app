import streamlit as st
import numpy as np
import cv2
from tensorflow import keras

# Page config
st.set_page_config(page_title="Theft Detection", page_icon="🚨")

st.title("🚨 Theft Detection System")
st.write("Upload an image to detect suspicious activity.")

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.55

# Load model (cached for speed)
@st.cache_resource
def load_model():
    return keras.models.load_model("theft_detection_model.keras")

model = load_model()

# Preprocessing
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# File upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    tensor = preprocess(img)
    prob = float(model.predict(tensor, verbose=0)[0][0])

    is_thief = prob >= CONFIDENCE_THRESHOLD
    label = "🚨 THIEF DETECTED" if is_thief else "✅ NORMAL"
    confidence = prob if is_thief else (1 - prob)

    st.subheader(f"Prediction: {label}")
    st.progress(int(confidence * 100))
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    if is_thief:
        st.error("⚠️ Suspicious activity detected!")
    else:
        st.success("All clear. No suspicious activity.")