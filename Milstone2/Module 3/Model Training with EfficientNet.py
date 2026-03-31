import torchvision.models as models
import torch.nn as nn
import torch
from datetime import datetime

# 1. Model Definition & Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# EfficientNet-B0 initialize
model = models.efficientnet_b0(weights='IMAGENET1K_V1') 


# Classes: Missing_hole, Mouse_bite, Normal, Open_circuit, Short, Spur, Spurious_copper
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7) 


model = model.to(device)

# Loss and Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Model successfully defined and ready for training!")

# ---------------------------------------------------------
# 2. Training Metrics & History Setup
history = {
    'train_loss': [], 
    'train_acc': [], 
    'val_loss': [], 
    'val_acc': []
}

# for Confusion Matrix
all_preds = []
all_labels = []

# ---------------------------------------------------------
# 3. Training Loop
num_epochs = 15
print("Training Started...")

for epoch in range(num_epochs):
    # --- TRAINING PHASE ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # --- VALIDATION PHASE ---
    model.eval()
    val_running_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
            # Final epoch mein confusion matrix ke liye labels collect karein
            if epoch == num_epochs - 1:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    val_loss = val_running_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    # Store Metrics in history dictionary
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ---------------------------------------------------------
# 4. Save Model

torch.save(model.state_dict(), 'pcb_final.pth') 
print("Model Weights Saved as 'pcb_final.pth'!")
