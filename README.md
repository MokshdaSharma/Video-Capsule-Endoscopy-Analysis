# *Video Capsule Endoscopy Analysis* 
AI-Powered Video Capsule Endoscopy Analysis: Detect abnormalities, generate reports, and assist doctors with automated insights.


### *Overview*

`Video Capsule Endoscopy (VCE) Analysis` is an AI-powered tool designed to detect abnormalities in endoscopic images using deep learning. It automates the diagnostic process, providing detailed medical insights and generating professional reports.

Traditional endoscopy requires manual analysis by medical professionals, which can be time-consuming and prone to human error. This project enhances diagnostic accuracy, speeds up the evaluation process, and ensures consistent analysis, making it a valuable tool for healthcare providers.

Users upload an endoscopic image through a web interface. The AI model, trained on medical datasets, classifies abnormalities with confidence scores. The system then retrieves detailed information, including potential causes, precautions, and treatment recommendations. Finally, it generates a downloadable PDF report for further medical use.

### *Features*
  - *Automated Endoscopy Analysis* – Upload an image for AI-based detection
  - *Accurate Abnormality Detection* – Deep learning model provides confidence scores
  - *Medical Insights* – Retrieve causes, precautions, and treatment recommendations
  - *PDF Report Generation* – Download detailed reports with diagnosis information
  - *Doctor Contact Integration* – Easily connect with medical professionals for further consultation

### *Model Architecture*
  - *Base Model*: `MobileNetV2` (pre-trained on ImageNet, fine-tuned for medical data)
  - *Head Layers*:
      - Global Average Pooling
      - Dense Layer (256 units, ReLU)
      - Output Layer (Softmax for multi-class classification)

### *How to Run*
To replicate the results, follow these steps:

1. Activate your virtual environment:
```bash
source env/Scripts/activate
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Download the trained model saved as:
```bash
disease_model.h5
```
4. Run the Streamlit application:
```bash
streamlit run app.py
```

### *🛠️Built With*
  - Python – Core programming language
  - TensorFlow/Keras – Deep learning model for classification
  - Streamlit – Interactive web-based user interface
  - OpenCV & PIL – Image processing libraries
  - FPDF – PDF report generation for medical documentation
  - pandas & NumPy – Data processing and analysis
  - Git & GitHub – Version control and collaboration

### *🔧Future Enhancements*
  - Implement real-time video analysis for continuous monitoring
  - Integrate telemedicine services for direct doctor consultations
