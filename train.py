import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from datasets.mhist import get_mhist_dataloader

# Configurations
CSV_PATH = "mhist_annotations.csv"
IMG_DIR = "images/"
BATCH_SIZE = 16
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/deeplabv3plus_resnet50.pth"

# Load Dataset
dataloaders = {
    "train": get_mhist_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, partition="train"),
    "val": get_mhist_dataloader(CSV_PATH, IMG_DIR, batch_size=BATCH_SIZE, partition="test"),
}

# Load Pretrained Model
model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=2)

if os.path.exists(MODEL_PATH):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=False)
    print("Loaded pretrained model from", MODEL_PATH)
else:
    print("Pretrained model not found, using default weights.")

model.classifier[4] = nn.Conv2d(256, 2, kernel_size=(1, 1))  # Adjust for 2 classes
model.to(DEVICE)

# Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train_model():
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in dataloaders[phase]:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)["out"]
                    labels = labels.Resize((16, 224, 224))
                    loss = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
    torch.save(model.state_dict(), "models/mhist_trained.pth")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train_model()
