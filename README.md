
# PCB Defect Detection & Classification System
An Automated Hybrid Approach using OpenCV & EfficientNet-B0

# Project Statement
The objective is to develop an automated system to detect and classify PCB defects (Missing hole, Mouse bite, Open circuit, Short, etc.) using reference-based image subtraction and CNN-based classification.

# System Architecture
The system follows a modular pipeline:

Image Preprocessing: Aligning template and test images using OpenCV.

Defect Localization: Subtraction and thresholding to find difference maps.

ROI Extraction: Contours are detected and individual defects are cropped.

AI Classification: EfficientNet-B0 classifies each crop into 7 categories.

Web Interface: Flask-based dashboard for real-time upload and reporting.

---
# Tech Stack
Image Processing: OpenCV, Numpy

Deep Learning: PyTorch, Torchvision (EfficientNet-B0)

Backend: Python, Flask, SQLAlchemy (PostgreSQL)

Frontend: HTML5, CSS3 (Responsive Design)

Dataset: DeepPCB

---

# Module-wise Features
🔹 Milestone 1: Image Processing
Subtraction Logic: Uses cv2.absdiff with a sensitive threshold of 20.

Noise Reduction: Gaussian Blur and Morphological operations to clean the mask.

Contour Detection: Extracts bounding boxes for every detected anomaly.

---
🔹 Milestone 2: AI Model
Model: EfficientNet-B0 (Transfer Learning).

Performance: High accuracy on validation set with a clear Confusion Matrix.

Input Size: 128x128 pixels.

---
🔹 Milestone 3: Web Integration
Dashboard: Shows real-time analysis results.

History: Users can view past scans stored in the database.

Authentication: Secure Login and Signup system.

---
# User Guide (How to Use)
Backend Setup

Clone the Repo:

Bash
git clone https://github.com/AlfiyaSaiyyad/PCB-Defect-Detection-and-Classification-System.git

cd PCB-Defect-Detection-and-Classification-System
Install Dependencies:

Bash
pip install -r requirements.txt
Database Configuration:
Update the SQLALCHEMY_DATABASE_URI in app.py with your PostgreSQL credentials.


Run Application:

Bash
python app.py

---

Frontend Usage

Login: Create an account and log in to the dashboard.

Upload: Go to the "Scan" page. Upload a PCB image.

Inference: The system will automatically compare it with the template in static/Template_images.

Results: View the labeled image with red bounding boxes and a detailed defect table.

Download: Use the "Download Full Report" button to save the result as a PDF (via window.print()).

---

# Evaluation Metrics

| Metric | Target Result |
| :--- | :--- |
| **Classification Accuracy** | ≥ 95%  Achieved |
| **Inference Time** | ≤ 3 sec Achieved |
| **Export Feature Working** | PDF/Image Export Fully Functional |
