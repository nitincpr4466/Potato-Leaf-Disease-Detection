
import streamlit as st

import tensorflow as tf
import numpy as np
from PIL import Image



# Load trained model
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "1" / "potato_leaf_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)


# Class names
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Page configuration
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered"
)

# App title
st.title("🥔 Potato Leaf Disease Detection")

st.write(
    "Upload a potato leaf image to detect whether "
    "the leaf is healthy or affected by a disease."
)
# Image uploader
uploaded_file = st.file_uploader(
    "Upload Potato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# Display uploaded image
if uploaded_file is not None:
    st.image(
        uploaded_file,
        caption="Uploaded Potato Leaf",
        use_container_width=True
    )



# Prediction button
if uploaded_file is not None:

    if st.button("Predict Disease"):

        # Read uploaded image
        img = Image.open(uploaded_file).convert("RGB")

        # Resize image
        img = img.resize((256, 256))

        # Convert image to array
        img_array = np.array(img)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)[0]

        # Class names for display
        display_names = [
            "Potato Early Blight",
            "Potato Late Blight",
            "Potato Healthy"
        ]

        # Get predicted class
        predicted_class = np.argmax(predictions)
        confidence = predictions[predicted_class]

        # Prediction result
        st.subheader("🔍 Prediction Result")

        if predicted_class == 2:
            st.success(f"🌿 {display_names[predicted_class]}")
        else:
            st.warning(f"⚠️ {display_names[predicted_class]} Detected")

        st.metric(
            "Model Confidence",
            f"{confidence * 100:.2f}%"
        )

        # Show class probabilities
        st.subheader("Class-wise Probabilities")

        for i in range(len(display_names)):

            probability = float(predictions[i])

            st.write(
                f"**{display_names[i]}: {probability * 100:.2f}%**"
            )

            st.progress(probability)
