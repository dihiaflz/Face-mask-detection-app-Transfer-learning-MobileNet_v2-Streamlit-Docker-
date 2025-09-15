import os
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
import tensorflow_hub as hub
from keras.layers import TFSMLayer



working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/final_model"
# Load the pre-trained model
model = TFSMLayer(model_path, call_endpoint="serving_default")

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(input_image_path):
    input_image = Image.open(input_image_path)

    input_image.show()

    input_image_recized = input_image.resize((224, 224))

    input_image_array = np.array(input_image_recized)

    input_image_scaled = input_image_array / 255.0

    input_image_reshaped = np.reshape(input_image_scaled, [1,224,224,3])

    return input_image_reshaped


# Function to Predict the Class of an Image
def predict_image_class(model, image_path):
    preprocessed_img = load_and_preprocess_image(image_path)

    input_prediction = model(preprocessed_img, training=False)

    # Some exported TensorFlow models (like TFSMLayer) return a dictionary instead of a plain tensor.
    # In that case, we need to extract the actual output tensor (usually the first value in the dict).
    if isinstance(input_prediction, dict):
        input_prediction = list(input_prediction.values())[0]

    # Convert tensor to NumPy array for further processing
    input_prediction = input_prediction.numpy()

    input_pred_label = np.argmax(input_prediction)

    if input_pred_label == 0:
        return('The person is not wearing a mask')
    else:
        return('The person is wearing a mask')

# Streamlit App
st.title('Face Mask Detector')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image)
            st.success(f'Prediction: {str(prediction)}')
