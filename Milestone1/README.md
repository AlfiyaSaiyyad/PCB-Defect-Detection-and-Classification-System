
# Milestone 1 — Dataset Preparation & Image Processing 

---

### Objective
The primary goal of this milestone is to convert raw PCB images into a structured dataset. We use **Classical Computer Vision** to detect defects via image subtraction and **XML Parsing** to extract precise Regions of Interest (ROIs) for future model training.

---

### Module 1: Image Subtraction & Defect Masking
In this module, we compare a "Template" (perfect PCB) with a "Test" (possibly defective PCB) to isolate differences.

#### Key Tasks & Logic:
* **Image Alignment:** Resizing test images to match template dimensions for pixel-to-pixel comparison.
* **CLAHE Enhancement:** Applied *Contrast Limited Adaptive Histogram Equalization* to improve visibility of fine traces.
* **Gaussian Blurring:** Noise reduction using a `(3, 3)` kernel before subtraction.
* **Absolute Difference:** Computing `cv2.absdiff()` to highlight deviations.
* **Morphological Operations:** Using `cv2.morphologyEx` (Opening) and `cv2.dilate` to clean up the binary mask and fill gaps in detected defects.



#### Deliverables:
* `ImageS.py`: Script for automated difference map generation.
* **Processed Outputs:** Defect-highlighted images categorized by type (Missing hole, Short, etc.).

---

### Module 2: XML-Driven ROI Extraction
Instead of manual cropping, we automate the generation of training samples using ground-truth annotations.

#### Key Tasks & Logic:
* **XML Parsing:** Using `xml.etree.ElementTree` to read defect coordinates (`xmin`, `ymin`, `xmax`, `ymax`) and labels.
* **Dynamic Cropping:** Extracting the exact bounding box from the high-resolution PCB images.
* **Smart Padding:** Adding a **5-pixel buffer** around defects to ensure the model sees enough context for classification.
* **Category Organization:** Automatically saving crops into a label-preserving folder structure (`PCB_ROI/Category_Name/`).



#### Deliverables:
* `ROI.py`: Automated pipeline for generating thousands of training patches.
* **Dataset:** A clean `PCB_ROI` folder with labeled defect samples (128x128 approx).

---

### Evaluation & Results
* **Defect Grading:** Defects are automatically categorized as **"Major"** (Area > 400px) or **"Minor"** (Area > 25px) based on contour analysis.
* **Accuracy:** High correlation between XML ground truth and extracted image patches.
* **Repeatability:** The script handles both `.jpg` and `.png` formats and safely skips missing files.

---

### Tech Stack
* **Language:** Python 3.x
* **Libraries:** OpenCV (Image Processing), NumPy (Array Math), XML.etree (Data Parsing)

---
**Author:** Alfiyabi Saiyyad
