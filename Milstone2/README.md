# Milestone 2 — Model Training & Evaluation 

---

### 🎯 Objective
The objective of this milestone is to implement a robust Deep Learning pipeline using **EfficientNet-B0** to classify PCB defects and evaluate the model's performance on unseen test data.

---

### 🧠 Module 3: Model Training with EfficientNet

#### 🛠️ Tasks Performed:
* **Neural Network Implementation:** Built a classification model using **PyTorch/Torchvision** with an EfficientNet-B0 backbone.
* **Image Preprocessing:** All defect images were resized to **128x128** pixels.
* **Data Augmentation:** Applied techniques like random flips, rotations, and color jittering to prevent overfitting.
* **Optimization Strategy:** Trained the model using the **Adam Optimizer** and **Cross-Entropy Loss** for 15+ epochs.

#### 📦 Deliverables:
* ✔️ **Trained Model:** Saved as `pcb_final.pth`.
* ✔️ **Metrics:** Training/Validation Accuracy and Loss logs.
* ✔️ **Visualizations:** Accuracy/Loss plots and a detailed **Confusion Matrix**.

---

### 🔍 Module 4: Evaluation & Prediction Testing

#### 🛠️ Tasks Performed:
* **Inference Pipeline:** Developed a script to run predictions on completely new, unseen PCB test images.
* **Ground Truth Validation:** Compared model predictions against the original annotated XML data.
* **Error Analysis:** Monitored False Positives and False Negatives to ensure high reliability.

#### 📦 Deliverables:
* ✔️ **Annotated Output:** Test images with predicted labels and confidence scores.
* ✔️ **Final Evaluation Report:** A comprehensive summary of precision, recall, and F1-score.

---

### 📊 Evaluation Benchmarks (Results)
* **Classification Accuracy:** Achieved **≥ 95%** accuracy on the test set.
* **Match Rate:** High correlation between predicted labels and ground-truth annotations.
* **Performance:** Stable and repeatable training curves indicating a well-generalized model.

---

### 💻 Tech Stack
* **Framework:** PyTorch (Deep Learning)
* **Vision:** OpenCV (Preprocessing)
* **Analysis:** Scikit-learn (Confusion Matrix & Reports)
* **Plotting:** Matplotlib & Seaborn

---
**Author:** Alfiyabi Saiyyad
