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

# Language selection dropdown (no typing option)
language = st.sidebar.selectbox("Choose Language / भाषा चुनें / భాషను ఎంచుకోండి",
                                options=["English", "Hindi", "Telugu"])

# Text content in different languages
content = {
    "English": {
        "title": "🌿 Plant Disease Detection",
        "description": "Identify plant diseases with high accuracy using AI-powered technology.",
        "about_header": "About",
        "about_content": "This app uses a Convolutional Neural Network (CNN) model to detect diseases in plant leaves. The model was trained on a large dataset of plant images and can accurately identify several types of plant diseases.",
        "how_to_use_header": "How to Use",
        "instructions": ["1. Upload a clear image of a plant leaf.",
                         "2. Wait for the model to process the image.",
                         "3. View the predicted disease name on the main screen."],
        "species_header": "Available Species",
        "species": [
            "1. Apple", "2. Blueberry", "3. Cherry", "4. Corn", "5. Grape",
            "6. Orange", "7. Peach", "8. Pepper", "9. Potato", "10. Raspberry",
            "11. Soybean", "12. Squash", "13. Strawberry", "14. Tomato"
        ],
        "upload_prompt": "Choose a leaf image...",
        "predicted": "Predicted Disease:",
        "analyzing": "Analyzing... Please wait",
        "footer": "Powered by QIS COLLEGE | Created by Kalyan Chakravarthy Pantham and his teammates"
    },
    "Hindi": {
        "title": "🌿 पौधों के रोग पहचान",
        "description": "एआई-संचालित तकनीक का उपयोग करके पौधों के रोगों की सटीक पहचान करें।",
        "about_header": "के बारे में",
        "about_content": "यह ऐप पौधों की पत्तियों में रोगों का पता लगाने के लिए एक कन्वोल्यूशनल न्यूरल नेटवर्क (CNN) मॉडल का उपयोग करता है।",
        "how_to_use_header": "कैसे इस्तेमाल करे",
        "instructions": ["1. पौधे की पत्ती की एक स्पष्ट छवि अपलोड करें।",
                         "2. मॉडल के छवि को प्रोसेस करने का इंतजार करें।",
                         "3. मुख्य स्क्रीन पर अनुमानित रोग का नाम देखें।"],
        "species_header": "उपलब्ध प्रजातियां",
        "species": [
            "1. सेब", "2. ब्लूबेरी", "3. चेरी", "4. मक्का", "5. अंगूर",
            "6. संतरा", "7. आड़ू", "8. मिर्च", "9. आलू", "10. रास्पबेरी",
            "11. सोयाबीन", "12. स्क्वैश", "13. स्ट्रॉबेरी", "14. टमाटर"
        ],
        "upload_prompt": "पत्ती की छवि चुनें...",
        "predicted": "अनुमानित रोग:",
        "analyzing": "विश्लेषण हो रहा है... कृपया प्रतीक्षा करें",
        "footer": "क्यूआईएस कॉलेज द्वारा संचालित | कल्याण चक्रवर्ती पंथम और उनकी टीम द्वारा निर्मित"
    },
    "Telugu": {
        "title": "🌿 మొక్కల వ్యాధి గుర్తింపు",
        "description": "AI ఆధారిత సాంకేతికతను ఉపయోగించి మొక్కల వ్యాధులను ఖచ్చితంగా గుర్తించండి.",
        "about_header": "గురించి",
        "about_content": "ఈ యాప్ మొక్కల ఆకుల్లో వ్యాధులను గుర్తించడానికి కన్‌వల్యూషనల్ న్యూరల్ నెట్‌వర్క్ (CNN) మోడల్‌ని ఉపయోగిస్తుంది.",
        "how_to_use_header": "వినియోగించడానికి ఎలా",
        "instructions": ["1. మొక్క ఆకుని స్పష్టమైన చిత్రం అప్‌లోడ్ చేయండి.",
                         "2. మోడల్ చిత్రాన్ని ప్రాసెస్ చేసే వరకు వేచి ఉండండి.",
                         "3. ప్రధాన స్క్రీన్‌లో అంచనా వ్యాధి పేరును చూడండి."],
        "species_header": "అందుబాటులో ఉన్న జాతులు",
        "species": [
            "1. ఆపిల్", "2. బ్లూబెర్రీ", "3. చెర్రీ", "4. మొక్కజొన్న", "5. ద్రాక్ష",
            "6. నారింజ", "7. పీచ్", "8. పెప్పర్", "9. బంగాళాదుంప", "10. రాస్ప్బెర్రీ",
            "11. సోయాబీన్", "12. స్క్వాష్", "13. స్ట్రాబెర్రీ", "14. టమోటా"
        ],
        "upload_prompt": "ఆకుని చిత్రం ఎంచుకోండి...",
        "predicted": "అంచనా వ్యాధి:",
        "analyzing": "విశ్లేషణ జరుగుతోంది... దయచేసి వేచి ఉండండి",
        "footer": "క్యూఐఎస్ కాలేజ్ ద్వారా నిర్వహించబడుతుంది | కల్యాణ్ చక్రవర్తి పంథం మరియు అతని టీమ్ ద్వారా సృష్టించబడింది"
    }
}

# Add a header and a logo/banner
st.image(
    "https://st4.depositphotos.com/1054144/24138/i/450/depositphotos_241389492-stock-photo-young-plant-in-sunlight.jpg",
    width=100)
st.title(content[language]["title"])
st.markdown(f"## {content[language]['description']}")

# Sidebar settings
st.sidebar.header(content[language]["about_header"])
st.sidebar.markdown(content[language]["about_content"])

st.sidebar.header(content[language]["how_to_use_header"])
# Display instructions in a vertical list with numbering
for instruction in content[language]["instructions"]:
    st.sidebar.markdown(f"- {instruction}")

st.sidebar.header(content[language]["species_header"])
# Display species in a vertical list with numbering
for species in content[language]["species"]:
    st.sidebar.markdown(f"- {species}")

# Upload an image file
uploaded_file = st.file_uploader(content[language]["upload_prompt"], type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=False, width=400)

    try:
        st.write(content[language]["analyzing"])
        model = tf.keras.models.load_model(output)

        def predict_image_class(model, image, class_indices):
            # Open and resize the image
            image = Image.open(image)
            image = image.resize((224, 224))  # Resize to model input size
            image = np.array(image) / 255.0  # Normalize to [0, 1]
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)
            return predicted_class

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
        predicted_class_name = class_indices.get(predicted_class[0], "Unknown Disease")

        st.write(f"**{content[language]['predicted']}** {predicted_class_name}")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Footer with dynamic content
st.markdown(f"""
<style>
footer {{
    position: fixed;
    bottom: 0;
    left: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: #333;
    text-align: center;
    padding: 10px;
}}
</style>
<footer>
    <p>{content[language]["footer"]}</p>
</footer>
""", unsafe_allow_html=True)

