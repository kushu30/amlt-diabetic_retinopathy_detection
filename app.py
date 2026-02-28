import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

# Page config
st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="retina",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Diabetic Retinopathy Early Detection System</h1>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.info("""
    This AI-powered system detects Diabetic Retinopathy from retinal fundus images.

    Severity Levels:
    - 0: No DR
    - 1: Mild NPDR
    - 2: Moderate NPDR
    - 3: Severe NPDR
    - 4: Proliferative DR
    """)

    st.title("How to use")
    st.write("1. Upload a retinal fundus image")
    st.write("2. Click 'Analyze Image'")
    st.write("3. View results and recommendations")


# Load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('diabetic_retinopathy_model.h5')
        return model
    except Exception:
        st.error("Model not found. Please run train.py first.")
        return None


# Preprocess image
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# Get prediction
def predict(model, image):
    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return predicted_class, confidence, predictions[0]


# Severity descriptions
severity_desc = {
    0: {"label": "No DR", "color": "green", "advice": "Regular check-ups recommended"},
    1: {"label": "Mild NPDR", "color": "blue", "advice": "Monitor blood sugar, annual eye exams"},
    2: {"label": "Moderate NPDR", "color": "orange", "advice": "Consult specialist, more frequent monitoring"},
    3: {"label": "Severe NPDR", "color": "red", "advice": "Urgent specialist consultation needed"},
    4: {"label": "Proliferative DR", "color": "darkred", "advice": "Immediate medical attention required"}
}

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Retinal Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear retinal fundus image"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image", type="primary"):
            with st.spinner("Analyzing..."):
                model = load_model()

                if model:
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)

                    predicted_class, confidence, all_probs = predict(model, image)

                    st.session_state['prediction'] = predicted_class
                    st.session_state['confidence'] = confidence
                    st.session_state['probabilities'] = all_probs

with col2:
    st.subheader("Analysis Results")

    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        conf = st.session_state['confidence']
        probs = st.session_state['probabilities']

        desc = severity_desc[pred]
        st.markdown(
            f"<div class='prediction-box' style='background-color: {desc['color']}20; border: 2px solid {desc['color']}'>"
            f"<h2 style='color: {desc['color']}'>{desc['label']}</h2>"
            f"<p>Confidence: {conf:.1%}</p>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.subheader("Severity Level")
        severity_label = f"Level {pred}"
        st.markdown(f"<h3>{severity_label}</h3>", unsafe_allow_html=True)

        st.subheader("Probability Distribution")
        fig, ax = plt.subplots()
        classes = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative']
        colors = ['green', 'blue', 'orange', 'red', 'darkred']
        bars = ax.bar(classes, probs, color=colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probability')
        ax.set_title('Class Probabilities')

        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{prob:.2%}', ha='center', va='bottom')

        st.pyplot(fig)

        st.subheader("Recommendations")
        st.info(desc['advice'])

        st.warning("This is an AI-assisted screening tool. Always consult with healthcare professionals.")

    else:
        st.info("Upload an image and click 'Analyze' to see results")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Developed for Diabetic Retinopathy Early Detection | Medical AI Assistant"
    "</p>",
    unsafe_allow_html=True
)
