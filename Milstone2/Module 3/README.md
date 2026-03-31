Module 3: Model Training & Evaluation (PCB Defect Classification)
This module covers the implementation, training, and evaluation of the Deep Learning architecture used to classify PCB defects. The model is designed to work as the "Brain" of the hybrid inspection system, categorizing localized patches into specific defect types.

Technical Implementation
The model was built using PyTorch and follows the specifications below:

Core Architecture: EfficientNet-B0

Input Dimensions: 128x128 pixels (RGB).

Classification Head: Custom Linear layer with 7 Output Nodes (6 Defect Categories + 1 Normal class).

Optimization Strategy:

Optimizer: Adam (Learning Rate: 0.001) for stable weight updates.

Loss Function: CrossEntropyLoss for multi-class classification.

Training Regime: 15 Epochs with real-time validation tracking.

Training Process & Metrics
The training was conducted with a clear split between training and validation sets to ensure the model generalizes well to unseen PCB scans.

Target Accuracy: ≥ 95% classification accuracy on the test/validation set.

Data Handling: Implemented automated label collection in the final epoch to generate a high-precision Confusion Matrix.

Evaluation Deliverables:
1. Accuracy & Loss Trends
The following plot illustrates the convergence of the model, showing a consistent decrease in loss and an increase in accuracy over 15 epochs.

2. Confusion Matrix
This matrix provides a detailed breakdown of the model's performance, confirming its ability to distinguish between similar defect classes like Spur and Short.

Dataset Categories
The model is trained to identify the following 7 classes:

Missing_hole

Mouse_bite

Normal

Open_circuit

Short

Spur

Spurious_copper

#### **1. Accuracy and Loss Curves**
![Accuracy Plot](Milestone_3/accuracy_plot.png) 

#### **2. Confusion Matrix**
![Confusion Matrix](Milestone_3/confusion_matrix.png)
