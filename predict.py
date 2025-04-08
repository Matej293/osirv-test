import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from network import modeling
from datasets.mhist import get_mhist_dataloader
from PIL import Image
import torchvision.transforms as T

# Function to parse command-line arguments
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--csv_path", type=str, default="mhist_annotations.csv",
                        help="Path to annotations CSV file")
    parser.add_argument("--img_dir", type=str, default="images",
                        help="Path to image directory")
    parser.add_argument("--model_path", type=str, default="models/deeplabv3plus_resnet50.pth",
                        help="Path to pre-trained model")
    parser.add_argument("--save_model_path", type=str, default="models/deeplabv3plus_resnet50_trained.pth",
                        help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training/testing")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID (set to -1 for CPU)")
    return parser

# Training function
def train_model(model, train_loader, device, epochs, save_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct, total = 0, 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            correct += (predicted == masks).sum().item()
            total += masks.numel()

        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    torch.save({'model_state': model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")

# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            predicted = torch.sigmoid(outputs) > 0.5
            correct += (predicted == masks).sum().item()
            total += masks.numel()

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

# Main function
def main():
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and opts.gpu_id != "-1" else "cpu")
    print(f"Using device: {device}")

    # Load Dataset
    train_loader = get_mhist_dataloader(opts.csv_path, opts.img_dir, opts.batch_size, partition="train")
    test_loader = get_mhist_dataloader(opts.csv_path, opts.img_dir, opts.batch_size, partition="test")

    # Load model
    NUM_CLASSES = 1
    model = modeling.deeplabv3plus_resnet50(num_classes=NUM_CLASSES, output_stride=16)

    # Modify classifier for binary classification
    model.classifier = modeling.DeepLabHeadV3Plus(
        in_channels=2048, low_level_channels=256, num_classes=NUM_CLASSES, aspp_dilate=[12, 24, 36]
    )

    for param in model.backbone.parameters():
        param.requires_grad = True

    # Load pre-trained weights (only for training, not for evaluation)
    if opts.mode == "train":
        # Train the model
        if os.path.exists(opts.model_path):
            checkpoint = torch.load(opts.model_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state']
            del state_dict['classifier.classifier.3.weight']
            del state_dict['classifier.classifier.3.bias']
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {opts.model_path}")
        
        model.to(device)
        train_model(model, train_loader, device, opts.epochs, opts.save_model_path)
    elif opts.mode == "eval":
        # For evaluation, load the newly trained model
        if os.path.exists(opts.save_model_path):
            checkpoint = torch.load(opts.save_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            print(f"Loaded trained model from {opts.save_model_path}")
        
        model.to(device)
        evaluate_model(model, test_loader, device)
    
if __name__ == "__main__":
    main()
