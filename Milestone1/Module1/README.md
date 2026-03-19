
# Module 1 — Image Preprocessing & Defect Detection (Image Subtraction)

---

## Objective

The objective of this module is to detect defects in Printed Circuit Board (PCB) images by comparing defected PCB images with defect-free reference templates using classical image processing techniques.

This module forms the **first stage** of the PCB Defect Detection System and prepares defect localization outputs for further ROI extraction and CNN-based classification.

---

## Methodology

The defect detection pipeline follows a reference-based comparison approach.

### 1️) Image Loading & Alignment

* Load **template (defect-free)** PCB image.
* Load **test PCB image** containing defects.
* Resize test image to match template dimensions for accurate comparison.

---

### 2️) Image Preprocessing

To enhance defect visibility and reduce noise:

* Convert images to grayscale.
* Apply **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to improve contrast.
* Apply **Gaussian Blur** to suppress noise and small intensity variations.

---

### 3️) Image Subtraction

The difference between template and test image is computed:

[
Difference = |Template - Test|
]

This highlights structural changes corresponding to PCB defects.

---

### 4️) Otsu Thresholding

Automatic thresholding is applied using **Otsu’s Method**:

* Determines optimal threshold automatically.
* Eliminates illumination dependency.
* Produces binary defect regions without manual tuning.

---

### 5️) Morphological Processing

To refine detected regions:

* Morphological Opening removes noise.
* Dilation strengthens defect areas.
* Improves contour detection accuracy.

---

### 6️) Defect Localization Using Contours

OpenCV contour detection is used to identify defect regions.

For each detected contour:

* Calculate contour area.
* Extract bounding box coordinates.
* Filter very small noise regions.

---

### 7️) Defect Annotation (Visualization Approach)

Instead of storing only binary masks, defects are visualized directly on PCB images:

* **Major Defect** → Large contour area(Green box)
* **Minor Defect** → Small contour area(red box)

Bounding boxes are drawn around detected defects for intuitive interpretation.

This visualization approach makes results easier for human inspection and presentation.

---

## Dataset Used

**DeepPCB Dataset**

Defect Categories:

* Missing_hole
* Mouse_bite
* Open_circuit
* Short
* Spur
* Spurious_copper

---

## Technologies & Libraries

* Python
* OpenCV
* NumPy

---

## Output Description

Annotated PCB images are generated containing detected defects.

Output Features:

* Automatic defect localization
* Colored bounding boxes
* Severity labeling (Minor / Major)
* Category-wise output folders

Output directory structure:

```bash
ImageSub_Outputs/
   Missing_hole/
   Mouse_bite/
   Open_circuit/
   Short/
   Spur/
   Spurious_copper/
```

Each output image contains highlighted defect regions.

---

## Sample Results

<img width="1594" height="792" alt="Screenshot 2026-02-22 000808" src="https://github.com/user-attachments/assets/24faeca7-4f86-4329-9d18-aef472013409" />

<img width="1431" height="746" alt="image" src="https://github.com/user-attachments/assets/5274d0bc-cb16-4586-a84e-2ee33e06f5ad" />


Example visual outputs:

* PCB image with detected defects
* Multiple defect localization using bounding boxes

---

## Evaluation Outcome

✔ Successful template-based defect comparison

✔ Robust defect localization

✔ Noise reduction achieved through preprocessing

✔ Clear visualization without relying only on binary masks

✔ Data prepared for ROI extraction (Module 2)

---




##  Next Module

Outputs generated in this module are used in:

 **Module 2 — Contour Detection & ROI Extraction**

where individual defect regions are cropped and prepared for CNN training.

---

## Author

Alfiyabi Saiyyad
