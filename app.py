import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

# Define the file ID
file_id = "1-025nCEuv7YsogDwFMNAf8ONrpUdYgZ7"

# Construct the full URL to the file
url = f"https://drive.google.com/uc?id={file_id}"

# Download the file
output = "plant_disease_prediction_model.h5"
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a header and a logo/banner
st.image(
    "https://st4.depositphotos.com/1054144/24138/i/450/depositphotos_241389492-stock-photo-young-plant-in-sunlight.jpg",
    width=100)
st.title("🌿 Plant Disease Detection")
st.markdown("## Identify plant diseases with high accuracy using AI-powered technology")

# Sidebar settings
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a Convolutional Neural Network (CNN) model to detect diseases in plant leaves.
The model was trained on a large dataset of plant images and can accurately identify several types of plant diseases.
""")

st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Upload a clear image of a plant leaf.
2. Wait for the model to process the image.
3. View the predicted disease name on the main screen.
""")

st.sidebar.header("Available Species")
st.sidebar.markdown("""
- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato
""")

# Upload an image file
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image (scaled down)
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=400)

    try:
        # Load the model and make predictions
        st.write("Analyzing... Please wait")
        model = tf.keras.models.load_model(output)

        def predict_image_class(model, image, class_indices):
            # Load and preprocess the image
            image = Image.open(image)
            image = image.resize((224, 224))  # Resize to model input size
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)
            return predicted_class

        # Map the predicted class to a disease name
        class_indices = {
            0: 'Apple___Apple_scab',
            1: 'Apple___Black_rot',
            2: 'Apple___Cedar_apple_rust',
            3: 'Apple___healthy',
            4: 'Blueberry___healthy',
            5: 'Cherry___Powdery_mildew',
            6: 'Cherry___healthy',
            7: 'Corn___Cercospora_leaf_spot',
            8: 'Corn___Common_rust',
            9: 'Corn___Northern_Leaf_Blight',
            10: 'Corn___healthy',
            11: 'Grape___Black_rot',
            12: 'Grape___Esca_(Black_Measles)',
            13: 'Grape___Leaf_blight',
            14: 'Grape___healthy',
            15: 'Orange___Citrus_greening',
            16: 'Peach___Bacterial_spot',
            17: 'Peach___healthy',
            18: 'Pepper___Bacterial_spot',
            19: 'Pepper___healthy',
            20: 'Potato___Early_blight',
            21: 'Potato___Late_blight',
            22: 'Potato___healthy',
            23: 'Raspberry___healthy',
            24: 'Soybean___healthy',
            25: 'Squash___Powdery_mildew',
            26: 'Strawberry___Leaf_scorch',
            27: 'Strawberry___healthy',
            28: 'Tomato___Bacterial_spot',
            29: 'Tomato___Early_blight',
            30: 'Tomato___Late_blight',
            31: 'Tomato___Leaf_Mold',
            32: 'Tomato___Septoria_leaf_spot',
            33: 'Tomato___Spider_mites',
            34: 'Tomato___Target_Spot',
            35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            36: 'Tomato___Tomato_mosaic_virus',
            37: 'Tomato___healthy'
        }
        predicted_class = predict_image_class(model, uploaded_file, class_indices)

        # Map the predicted class to a disease name
        predicted_class_name = class_indices.get(predicted_class[0], "Unknown Disease")

        # Display the prediction result
        st.write(f"**Predicted Disease:** {predicted_class_name}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer with CSS
st.markdown("""
<style>
footer {
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #333;
    text-align: center;
    padding: 10px;
}
</style>
<footer>
    <p>Powered by QIS COLLEGE | Created by kalyan chakravarthy pantham and his teammates</p>
</footer>
""", unsafe_allow_html=True)
