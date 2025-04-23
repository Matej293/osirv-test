import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasets.mhist import get_mhist_dataloader

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to MHIST CSV', default='./data/mhist_annotations.csv')
    parser.add_argument('--img_dir', type=str, help='Directory with MHIST images', default='./data/images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default='./models/classifier_resnet50.pth')
    parser.add_argument('--gpu_id', type=str, default="0")
    return parser.parse_args()

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === DATASET ===
    # Define augmentation configuration
    train_augmentation_config = {
        'resize': (256, 256),
        'crop': (224, 224),
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'rotation_degrees': 15,
        'brightness': 1.2,
        'contrast': 1.5,
        'saturation': 0.1,
        'hue': 0.0,
        'translate': (0.1, 0.1)
    }
    
    test_augmentation_config = {
        'resize': (224, 224)
    }
    
    # Use the unified dataloader function
    train_loader = get_mhist_dataloader(
        csv_file=args.csv_path, 
        img_dir=args.img_dir, 
        batch_size=args.batch_size, 
        partition="train", 
        augmentation_config=train_augmentation_config,
        task="classification"
    )
    
    val_loader = get_mhist_dataloader(
        csv_file=args.csv_path, 
        img_dir=args.img_dir, 
        batch_size=args.batch_size, 
        partition="test", 
        augmentation_config=test_augmentation_config,
        task="classification"
    )

    # === MODEL ===
    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    lr = 0.00001572
    weight_decay = 0.000010
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # === TRAINING LOOP ===
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

        # validation (optional)
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Validation Acc: {val_acc:.4f}")

    # === SAVE MODEL ===
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()
