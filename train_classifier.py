import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from datasets.mhist import get_mhist_dataloader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, inputs, targets):
        loss_dice = self.dice(inputs, targets)
        loss_bce = self.bce(inputs, targets)
        return self.weight_dice * loss_dice + self.weight_bce * loss_bce

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to MHIST CSV', default='./data/mhist_annotations.csv')
    parser.add_argument('--img_dir', type=str, help='Directory with MHIST images', default='./data/images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--save_path', type=str, default='./models/classifier_resnet34.pth')
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
        'brightness': 0.2,
        'contrast': 0.2,
        'saturation': 0.1,
        'hue': 0.05,
        'translate': (0.1, 0.1)
    }
    
    test_augmentation_config = {
        'resize': (224, 224)
    }
    
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
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 1)
    )
    model = model.to(device)

    config = {
        'data.ssa_count': 990,
        'data.hp_count': 2162,
    }

    # weighted loss
    class_weight = (config.get('data.ssa_count') + config.get('data.hp_count')) / config.get('data.ssa_count')
    criterion = CombinedLoss(weight_dice=1.0, weight_bce=1.0, pos_weight=torch.tensor([class_weight], dtype=torch.float).to(device))

    lr = args.lr
    weight_decay = 0.01
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    best_val_acc = 0.0
    
    # === TRAINING LOOP ===
    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        y_true, y_pred = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            
            labels_float = labels.float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_float)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            running_loss += loss.item()
            
            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels_float).sum().item()
            total += labels.size(0)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().view(-1).numpy())

        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

        if epoch % 5 == 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )
            print(f"Train Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        model.eval()
        correct, total = 0, 0
        val_y_true, val_y_pred, val_outputs_all = [], [], []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                labels_float = labels.float().unsqueeze(1)
                
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).int()
                
                correct += (preds == labels_float).sum().item()
                total += labels.size(0)
                
                val_y_true.extend(labels.cpu().numpy())
                val_y_pred.extend(preds.cpu().view(-1).numpy())
                val_outputs_all.extend(torch.sigmoid(outputs).cpu().view(-1).numpy())
        
        val_acc = correct / total if total > 0 else 0
        print(f"Validation Acc: {val_acc:.4f}")
        
        if epoch % 5 == 0 and len(set(val_y_true)) > 1:
            try:
                val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                    val_y_true, val_y_pred, average='binary', zero_division=0
                )
                val_auc = roc_auc_score(val_y_true, val_outputs_all)
                print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
            except:
                print("Couldn't calculate validation metrics")
        
        scheduler.step()
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, args.save_path)
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to {args.save_path}")

if __name__ == "__main__":
    main()
