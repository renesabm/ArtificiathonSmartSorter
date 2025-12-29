import streamlit as st
import numpy as np
from PIL import Image
import requests
import tensorflow as tf
import time

@st.cache_resource
def load_model():
    model_path = "artificiationmobilenetv2_battery.py"
    return tf.keras.models.load_model(
        model_path,
        compile=False
    )
model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0

def start_over():
    st.session_state.uploader_key += 1
    st.session_state.camera_key += 1
    st.rerun()
st.title("The SortSmart Waste Detector")
st.subheader("to make your sorting an easier and more sustainable process")
option = st.radio("Select option:", ["Upload Image", "Use Camera"])
st.write("Make sure inputted image is clear and shows the entire object")
image = None

class_names = ["compost", "landfill","recycle","toxic"]

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        key=f"uploader_{st.session_state.uploader_key}"
    )
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Use Camera":
    camera_image = st.camera_input(
        "Take a picture",
        key = f"camera_{st.session_state.uploader_key}"
    )
    if camera_image:
        image = Image.open(camera_image)
if image:
    st.session_state.processed = True
    st.image(image, caption="Selected Image", use_column_width=True)

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Preprocessing image...")
    processed_image = preprocess_image(image)
    progress_bar.progress(20)

    status_text.text("Running model prediction...")
    predictions = model.predict(processed_image)
    progress_bar.progress(40)
    status_text.text("Running model prediction...")
    progress_bar.progress(60)
    status_text.text("Running model prediction...")
    progress_bar.progress(80)

    predicted_class = np.argmax(predictions)
    progress_bar.progress(100)
    status_text.text("Operation complete!")

    st.success(f"Prediction: {class_names[predicted_class]}")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    st.button("Classify another image", on_click=start_over)





