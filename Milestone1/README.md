# PCB-Defect-Detection-and-Classification-System
An automated PCB defect detection and classification system using Image Processing (OpenCV) and Deep Learning (CNN). Features reference-based subtraction, ROI extraction via XML parsing, and a planned web frontend for real-time defect labeling.



###  Project Objective
The objective is to develop an automated defect detection and classification system for Printed Circuit Boards (PCBs) using **Image Processing** and **Deep Learning** techniques. The system employs reference-based image subtraction, contour extraction, and **CNN-based classification** to identify and label defects.

---

### Milestone 1 : 
I have successfully completed the Image Substraction and ROI Extraction phase.

#### Key Achievements:
* **Image Subtraction:** Used `cv2.absdiff` and **Otsu Thresholding** to highlight defects.
* **XML Parsing:** Automated the extraction of defect coordinates from annotation files.
* **ROI Extraction:** Developed a script to crop defect regions with padding for high-quality training data.
* **Automatic Labeling:** Organized crops into 6 categories (Short, Open, Missing Hole, etc.) using folder-based labeling.

---

### Tech Stack
* **Language:** Python 3.x
* **Libraries:** OpenCV, NumPy, XML.etree


---

### Repository Structure
* `Code/`: Contains Python scripts for ROI extraction and Image Processing.
* `Results/`: Sample images showing detected defects with bounding boxes.
* `README.md`: Project documentation.
* `.gitignore`: To prevent uploading heavy dataset files.

---

### How to Run
1. Clone the repository.
2. Install dependencies: `pip install opencv-python numpy`.
3. Run the extraction script: `python Code/roi_extraction.py`.
