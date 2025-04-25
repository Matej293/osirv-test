import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from datasets.mhist import get_mhist_dataloader
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, help='Path to MHIST CSV', default='./data/mhist_annotations.csv')
    parser.add_argument('--img_dir', type=str, help='Directory with MHIST images', default='./data/images')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--save_path', type=str, default='./models/classifier_resnet34_trained.pth')
    parser.add_argument('--gpu_id', type=str, default="0")
    return parser.parse_args()

def freeze_layers(model, unfreeze_layers=None):
    """Freeze all layers except those specified"""
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    for param in model.fc.parameters():
        param.requires_grad = True
        
    if unfreeze_layers is None:
        unfreeze_layers = []
    elif isinstance(unfreeze_layers, str):
        unfreeze_layers = [unfreeze_layers]
        
    for layer in unfreeze_layers:
        for name, param in model.named_parameters():
            if layer in name:
                param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} out of {total_params:,} ({trainable_params/total_params:.2%})")

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === DATASET ===
    train_augmentation_config = {
        'resize': (256, 256),
        'crop': (224, 224),
        'horizontal_flip_prob': 0.5,
        'vertical_flip_prob': 0.5,
        'rotation_degrees': 30,
        'brightness': 1.2,
        'contrast': 1.5,
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

    criterion = FocalLoss(alpha=0.75, gamma=1.0)
    
    # === GRADUAL UNFREEZING SCHEDULE ===
    unfreezing_schedule = [
        {"epochs": 5, "layers": None},
        {"epochs": 5, "layers": "layer4"},
        {"epochs": 5, "layers": "layer3"},
        {"epochs": 5, "layers": "layer2"},
        {"epochs": 10, "layers": "layer1"}
    ]
    
    # For early stopping
    patience = 10
    best_val_acc = 0.0
    early_stop_counter = 0
    
    epoch_counter = 0
    
    # === TRAINING LOOP WITH GRADUAL UNFREEZING ===
    accumulated_layers = []
    for phase_idx, phase in enumerate(unfreezing_schedule):
        if phase["layers"]:
            accumulated_layers.append(phase["layers"])
        
        print(f"\n=== Phase {phase_idx+1}: Unfreezing {phase['layers'] if phase['layers'] else 'None (only FC)'} ===")
        
        freeze_layers(model, accumulated_layers)
        
        lr = args.lr / (2 ** phase_idx)
        print(f"Learning rate: {lr}")
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3,
        )
        
        for epoch in range(phase["epochs"]):
            epoch_counter += 1
            model.train()
            running_loss, correct, total = 0.0, 0, 0
            y_true, y_pred = [], []

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch_counter}/{sum(p['epochs'] for p in unfreezing_schedule)}"):
                imgs, labels = imgs.to(device), labels.to(device)
                
                labels_float = labels.float().unsqueeze(1)

                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels_float)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                running_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.35).int()
                correct += (preds == labels_float).sum().item()
                total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().view(-1).numpy())

            train_acc = correct / total
            print(f"[Epoch {epoch_counter}] Loss: {running_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validation
            model.eval()
            correct, total = 0, 0
            val_y_true, val_y_pred, val_outputs_all = [], [], []
            val_loss = 0.0
            
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    labels_float = labels.float().unsqueeze(1)
                    
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_float)
                    val_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.35).int()
                    
                    correct += (preds == labels_float).sum().item()
                    total += labels.size(0)
                    
                    val_y_true.extend(labels.cpu().numpy())
                    val_y_pred.extend(preds.cpu().view(-1).numpy())
                    val_outputs_all.extend(torch.sigmoid(outputs).cpu().view(-1).numpy())
            
            val_acc = correct / total if total > 0 else 0
            print(f"Validation Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")
            
            # calculate additional metrics periodically
            if len(set(val_y_true)) > 1:
                try:
                    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
                        val_y_true, val_y_pred, average='binary', zero_division=0
                    )
                    val_auc = roc_auc_score(val_y_true, val_outputs_all)
                    print(f"Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
                except:
                    print("Couldn't calculate validation metrics")
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                torch.save({
                    'epoch': epoch_counter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, args.save_path)
                print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch_counter} epochs")
                break
        
        if early_stop_counter >= patience:
            break

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to {args.save_path}")

if __name__ == "__main__":
    main()
