import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import time
from fpdf import FPDF
import docx  

# Load model
model = tf.keras.models.load_model(r'C:\Users\Mokshda Sharma\Desktop\My Projects\VCE_Analysis\Capsule_vision\disease_model.h5')

# Define class labels
class_labels = [
    "Angioectasia", "Bleeding", "Erosion", "Erythema", "Foreign body",
    "Lymphangiectasia", "Polyp", "Ulcer", "Worms", "Normal"
]

# Function to read disease info 
def fetch_medical_info(disease_name):
    df = pd.read_csv(r'C:\Users\Mokshda Sharma\Desktop\My Projects\Capsule vision\VCE_abnormality_info.csv')
    row = df[df["Disease"].str.lower() == disease_name.lower()]
    if not row.empty:
        return {
            "cause": row["Causes"].values[0],
            "precautions": row["Precautions"].values[0],
            "treatment": row["Treatment"].values[0]
        }
    return {"cause": "N/A", "precautions": "N/A", "treatment": "N/A"}


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))  
    image = np.array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

# Function to generate a PDF report
def generate_report(predicted_class, confidence, info):
    pdf = FPDF()
    pdf.add_page()
    
    # Use a font that supports Unicode (DejaVuSans supports most characters)
    pdf.add_font("Arial", "", r"C:\Windows\Fonts\arial.ttf", uni=True)  # Load system Arial font
    pdf.set_font("Arial", "B", 16)
    
    pdf.cell(200, 10, "Medical Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, f"Detected Abnormality: {predicted_class}", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(200, 10, f"Confidence: {confidence:.2f}%", ln=True)
    pdf.ln(5)

    # Ensure special characters are handled
    cause_text = info['cause'].encode('latin-1', 'replace').decode('latin-1')
    precautions_text = info['precautions'].encode('latin-1', 'replace').decode('latin-1')
    treatment_text = info['treatment'].encode('latin-1', 'replace').decode('latin-1')

    pdf.multi_cell(0, 10, f"Possible Cause:\n{cause_text}")
    pdf.multi_cell(0, 10, f"Precautions:\n{precautions_text}")
    pdf.multi_cell(0, 10, f"Treatment:\n{treatment_text}")

    filename = "Medical_Report.pdf"
    pdf.output(filename, "F")  # "F" forces file output, ensuring encoding compatibility
    return filename


# Streamlit UI
st.set_page_config(page_title="VCE Abnormality Detection", layout="wide")

st.title("🔬 Video Capsule Endoscopy AI")
st.write("Upload an endoscopy image to detect abnormalities and generate a medical report.")

# Sidebar for interactivity
st.sidebar.image(r"C:\Users\Mokshda Sharma\Desktop\My Projects\Capsule vision\logo.png") 
st.sidebar.header("🔍 How It Works")
st.sidebar.write("""
1. Upload an image from your device or webcam  
2. AI model will analyze the image  
3. You’ll get the detected abnormality with confidence score  
4. Download a medical report with diagnosis details  
""")

# Sidebar for doctor details
st.sidebar.header("👨‍⚕️ Contact a Doctor")
st.sidebar.image(r"C:\Users\Mokshda Sharma\Desktop\My Projects\Capsule vision\doctor.png") 

st.sidebar.write("**Dr. John Doe**")
st.sidebar.write("📍 **Clinic Address:** 123 Health Street, City")
st.sidebar.write("📞 **Phone:** +1 234-567-8901")
st.sidebar.write("📧 **Email:** dr.johndoe@example.com")
st.sidebar.write("🕒 **Timings:** Mon-Fri (9 AM - 5 PM)")

st.sidebar.markdown("---")
st.sidebar.write("For emergencies, visit your nearest hospital.")


# File uploader & webcam support
uploaded_file = st.file_uploader("📷 Upload an image...", type=["jpg", "png", "jpeg"])
use_webcam = st.button("📸 Capture from Webcam")

if use_webcam:
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Processing & Prediction
    with st.spinner("🧠 Analyzing image... Please wait."):
        time.sleep(2)  # Simulating processing time
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

    # Get prediction result
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100  

    # Fetch disease info from API
    info = fetch_medical_info(predicted_class)

    # Display result
    st.success("✅ Analysis Complete!")
    st.subheader("🩺 Prediction Result:")
    st.write(f"**Detected Abnormality:** `{predicted_class}`")

    # Show disease information
    with st.expander("📌 More Info"):
        st.write(f"**Possible Cause:**\n{info['cause']}")
        st.write(f"**Precautions:**\n{info['precautions']}")
        st.write(f"**Treatment:**\n{info['treatment']}")

    # Generate & Download Report
    report_file = generate_report(predicted_class, confidence, info)
    with open(report_file, "rb") as file:
        st.download_button("📄 Download Medical Report", file, file_name="Medical_Report.pdf", mime="application/pdf")
