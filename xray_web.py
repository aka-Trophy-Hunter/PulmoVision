import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/xray_model.hdf5")
    return model

with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # PulmoVision
         """
         )

file = st.file_uploader("Please upload a chest scan file", type=["jpg", "jpeg", "png"])


def import_and_predict(image_data, model):
    size = (180, 180)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)

    st.subheader(f"Result: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")

    if predicted_class == "PNEUMONIA":
        st.warning("This scan shows signs consistent with pneumonia. Please consult a doctor for confirmation.")
    else:
        st.success("This scan appears normal.")

    st.caption("This is an automated screening tool and not a substitute for professional medical diagnosis.")
