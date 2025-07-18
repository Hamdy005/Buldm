import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from keras.applications.resnet import preprocess_input
from pathlib import Path

# Constants for Buldm app
IMAGE_SIZE = 224
CLASSES = ['handbag', 'id', 'keys', 'phone', 'wallet']
MODEL_PATH = Path(__file__).resolve().parent / "Models" / "buldm.h5"

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_image(image, model):
    if model is None:
        return None, 0.0

    # Preprocess image
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    pred = model.predict(img_array, verbose=0)
    confidence = np.max(pred[0])
    entropy = -np.sum(pred[0] * np.log(pred[0] + 1e-10))

    CONFIDENCE_THRESHOLD = 0.5
    ENTROPY_THRESHOLD = 1.5

    if confidence < CONFIDENCE_THRESHOLD or entropy > ENTROPY_THRESHOLD:
        return None, confidence

    class_idx = np.argmax(pred[0])
    class_name = CLASSES[class_idx]
    return class_name, confidence

def main():
    st.title("👜 Buldm - Personal Item Classifier")
    st.markdown("Upload an image of a personal item (handbag, ID, keys, phone, wallet) to classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    model = load_model()

    if uploaded_file is not None and model is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        class_name, confidence = predict_image(image, model)

        st.image(image, channels="BGR", caption="Uploaded Image")
        st.header("🔎 Prediction Results")

        if class_name is None:
            st.warning("⚠️ Unknown or unclear item!")
            st.write("This item doesn't match any of the known categories (handbag, ID, keys, phone, wallet).")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Detected Item", class_name.title())
            with col2:
                st.metric("Confidence", f"{confidence:.2%}")

            # Confidence bar
            if confidence > 0.8:
                st.success(f"High confidence: {confidence:.2%}")
            elif confidence > 0.5:
                st.warning(f"Moderate confidence: {confidence:.2%}")
            else:
                st.error(f"Low confidence: {confidence:.2%}")

            st.progress(min(max(float(confidence), 0.0), 1.0))

if __name__ == "__main__":
    main()