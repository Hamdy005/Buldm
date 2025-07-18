import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.applications.resnet import preprocess_input
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Buldm - Personal Item Classifier",
    page_icon="👜",
    layout="centered"
)

# Constants
IMAGE_SIZE = 224
CLASSES = ['handbag', 'id', 'keys', 'phone', 'wallet']
MODEL_PATH = Path(__file__).resolve().parent / "Models" / "buldm.h5"

@st.cache_resource
def load_model():
    try:
        # Load model WITHOUT custom options
        model = keras.models.load_model(
            MODEL_PATH,
            compile=False
        )
        # Recompile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def predict_image(image, model):
    if model is None:
        return None, None
    
    # Preprocess image
    img = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)
    
    # Make prediction
    pred = model.predict(img_array, verbose=0)
    confidence = np.max(pred[0])
    class_idx = np.argmax(pred[0])
    
    # Get class name and confidence
    class_name = CLASSES[class_idx]
    return class_name, confidence

def main():
    # Page header
    st.title("👜 Buldm - Personal Item Classifier")
    st.markdown("""
    This app helps you identify common personal items like handbags, IDs, keys, phones, and wallets.
    Simply upload an image to get started!
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    if model is None:
        st.error("❌ Failed to load model. Please try again later.")
        return
        
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a personal item"
    )
    
    if uploaded_file:
        try:
            # Load and display image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Create columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, channels="BGR", caption="Uploaded Image")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                class_name, confidence = predict_image(image, model)
            
            # Display results
            with col2:
                st.markdown("### Results")
                if confidence > 0.7:
                    st.success(f"**Detected Item:** {class_name.title()}")
                else:
                    st.warning(f"**Possible Item:** {class_name.title()}")
                
                # Confidence display
                confidence_pct = float(confidence)
                st.progress(confidence_pct)
                st.markdown(f"**Confidence:** {confidence_pct:.1%}")
                
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()