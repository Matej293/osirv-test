import os
import argparse
import tempfile
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from network import modeling

from metrics.wandb_logger import WandbLogger
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from utils.visualization import visualize_segmentation_results

# parsing command-line arguments
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"],
                        help="Mode: train or eval (if not provided, will do both)")
    parser.add_argument("--csv_path", type=str,
                        help="Path to annotations CSV file")
    parser.add_argument("--img_dir", type=str,
                        help="Path to image directory")
    parser.add_argument("--model_path", type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--save_model_path", type=str,
                        help="Path to save trained model")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training/testing")
    parser.add_argument("--epochs", type=int,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float,
                      help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float,
                      help="Weight decay for optimizer")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID to use for training/testing")
    parser.add_argument("--threshold", type=float,
                      help="Classification threshold")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="mhist-classification-5",
                      help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                      help="WandB run name")
    parser.add_argument("--init_timeout", type=int, default=180,
                      help="Timeout for wandb initialization")
    return parser

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou

class CombinedLoss(nn.Module):
    def __init__(self, weight_dice=1.0, weight_bce=1.0, weight_iou=1.0, pos_weight=None):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.weight_iou = weight_iou
        self.dice = DiceLoss()
        self.iou = IoULoss()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    def forward(self, inputs, targets):
        loss_dice = self.dice(inputs, targets)
        loss_bce = self.bce(inputs, targets)
        loss_iou = self.iou(inputs, targets)
        return self.weight_dice * loss_dice + self.weight_bce * loss_bce + self.weight_iou * loss_iou

# Training function
def train_model(model, train_loader, device, config, logger=None, save_path=None, val_loader=None):
    """Train the model with the given configuration."""
    class_weight = (config.get('data.ssa_count') + config.get('data.hp_count')) / config.get('data.ssa_count')
    criterion = CombinedLoss(
        weight_dice=0.6, 
        weight_bce=0.2, 
        weight_iou=1.0,
        pos_weight=torch.tensor([class_weight], dtype=torch.float).to(device)
    ) 
    lr = float(config.get('training.learning_rate'))
    weight_decay = float(config.get('training.weight_decay'))
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.get('training.epochs'),
        eta_min=0.0,
    )
    
    # Training loop
    model.train()
    epochs = config.get('training.epochs')
    log_interval = config.get('logging.batch_log_interval')
    threshold = config.get('training.threshold')
    
    # early stopping parameters
    patience = config.get('training.patience', 10)
    early_stop_counter = 0
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct, total = 0, 0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # applying thresholds for metrics
            batch_loss = loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > threshold).float()
            
            # updating batch metrics
            batch_correct = (predicted == masks).sum().item()
            batch_total = masks.numel()
            
            epoch_loss += batch_loss
            correct += batch_correct
            total += batch_total
            
            # logging batch results
            if logger and batch_idx % log_interval == 0:
                batch_accuracy = batch_correct / batch_total
                
                train_step = (epoch + 1) * len(train_loader) + batch_idx
                
                logger.log_scalar('Train/BatchLoss', batch_loss, step=train_step)
                logger.log_scalar('Train/BatchAccuracy', batch_accuracy, step=train_step)

        # epoch metrics
        epoch_loss /= len(train_loader)
        accuracy = correct / total

        logger.log_scalar('Train/EpochLoss', epoch_loss, step=epoch + 1)
        logger.log_scalar('Train/Accuracy', accuracy, step=epoch + 1)
        logger.log_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], step=epoch + 1)
        logger.log_scalar('Train/WeightDecay', optimizer.param_groups[0]['weight_decay'], step=epoch + 1)

        # validation phase
        if val_loader is not None:
            model.eval()
            val_correct, val_total = 0, 0
            val_y_true, val_y_pred, val_outputs_all = [], [], []
            val_loss = 0.0
            
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)
                    masks = masks.unsqueeze(1).float()
                    
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    predicted = (probs > threshold).float()
                    
                    val_correct += (predicted == masks).sum().item()
                    val_total += masks.numel()
                    
                    val_y_true.extend(masks.cpu().numpy().flatten())
                    val_y_pred.extend(predicted.cpu().numpy().flatten())
                    val_outputs_all.extend(probs.cpu().numpy().flatten())
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            val_loss /= len(val_loader)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
                  f"Accuracy: {accuracy:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            if logger:
                logger.log_scalar('Validation/Accuracy', val_acc, step=epoch + 1)
                logger.log_scalar('Validation/Loss', val_loss, step=epoch + 1)
            
            # early stopping check and model saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop_counter = 0
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    torch.save({'model_state': model.state_dict()}, save_path)
                    print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                break
                
            model.train()
        else:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        scheduler.step()

    if val_loader is not None:
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Final model saved to {save_path}")

    return model

# Evaluation function
def evaluate_model(model, test_loader, device, config, logger, step=None):
    model.eval()
    class_weight = (config.get('data.ssa_count') + config.get('data.hp_count')) / config.get('data.ssa_count')
    criterion = CombinedLoss(
        weight_dice=0.6, 
        weight_bce=0.2, 
        weight_iou=1.0,
        pos_weight=torch.tensor([class_weight], dtype=torch.float).to(device)
    )
    
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_targets = [], []
    threshold = config.get('training.threshold')

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            
            predicted = (probs > threshold).float()
            
            correct += (predicted == masks).sum().item()
            total += masks.numel()
           
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    # logging evaluation metrics
    metrics = logger.log_evaluation_metrics(all_preds, all_targets, accuracy=accuracy, avg_loss=avg_loss, step=step)
    
    print(f"Evaluation — Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}, IoU: {metrics['iou']:.4f}")

    # logging sample images
    if len(test_loader) > 0:
        try:
            sample_images, sample_masks = next(iter(test_loader))
            sample_images = sample_images.to(device)
            sample_masks = sample_masks.unsqueeze(1).float().to(device)
            
            with torch.no_grad():
                sample_outputs = model(sample_images)
                sample_probs = torch.sigmoid(sample_outputs)
                sample_preds = (sample_probs > threshold).float()

                visualize_segmentation_results(
                    images=sample_images,
                    masks=sample_masks,
                    predictions=sample_preds,
                    probabilities=sample_probs,
                    step=step,
                    logger=logger
                )
        except Exception as e:
            print(f"Warning: Could not visualize segmentation results: {e}")
    
    return metrics

# Main function
def main():
    parser = get_argparser()
    args = parser.parse_args()
    
    # load config
    config = ConfigManager(args.config)
    config.update_from_args(args)
    
    # device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    wandb_config = {
        "model": config.get("model"),
        "data": config.get("data"),
        "training": config.get("training"),
        "augmentation": config.get("augmentation"),
    }
    
    logger = WandbLogger(
        project=args.wandb_project,
        name=args.wandb_name,
        config=wandb_config
    )
    print("Using Weights & Biases for logging")

    # loader setup
    train_loader = get_mhist_dataloader(
        csv_file=config.get('data.csv_path'), 
        img_dir=config.get('data.img_dir'), 
        mask_dir=config.get('data.mask_dir'),
        batch_size=config.get('data.batch_size'),
        partition="train",
        augmentation_config=config.get('augmentation.train'),
        task="segmentation"
    )
    
    test_loader = get_mhist_dataloader(
        csv_file=config.get('data.csv_path'), 
        img_dir=config.get('data.img_dir'), 
        mask_dir=config.get('data.mask_dir'),
        batch_size=config.get('data.batch_size'),
        partition="test",
        augmentation_config=config.get('augmentation.test'),
        task="segmentation"
    )

    # model setup
    model = modeling.deeplabv3plus_resnet101(
        num_classes=config.get('model.num_classes'),
        output_stride=config.get('model.output_stride'),
        pretrained_backbone=config.get('model.pretrained_backbone')
    )

    # if train/eval mode is not provided, run both train and eval
    run_train = args.mode is None or args.mode == "train"
    run_eval = args.mode is None or args.mode == "eval"

    # Training phase
    if run_train:
        model_path = config.get('model.pretrained_path')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                if 'model_state' in checkpoint:
                    state_dict = checkpoint['model_state']
                    if 'classifier.classifier.3.weight' in state_dict:
                        del state_dict['classifier.classifier.3.weight']
                    if 'classifier.classifier.3.bias' in state_dict:
                        del state_dict['classifier.classifier.3.bias']
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded model from {model_path}")
                else:
                    print(f"Warning: Invalid checkpoint format in {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
            
        model.classifier = modeling.DeepLabHeadV3Plus(
            in_channels=2048,
            low_level_channels=256, 
            num_classes=config.get('model.num_classes'),
            aspp_dilate=config.get('model.aspp_dilate')
        )

        # print the hyperparameters
        print("\nTraining with hyperparameters:")
        print(f"Learning Rate: {config.get('training.learning_rate')}")
        print(f"Weight Decay: {config.get('training.weight_decay')}")
        print(f"Epochs: {config.get('training.epochs')}")
        print(f"Batch Size: {config.get('data.batch_size')}")
        print(f"Threshold: {config.get('training.threshold')}")
        print("\nAugmentation parameters:")
        for key, value in config.get('augmentation.train').items():
            print(f"{key}: {value}")

        model.to(device)
        trained_model = train_model(
            model, 
            train_loader, 
            device, 
            config,
            logger,
            config.get('model.saved_path'),
            val_loader=test_loader
        )
        
        if run_eval and args.mode is None:
            print("\n--- Running evaluation after training ---\n")
            evaluate_model(trained_model, test_loader, device, config, logger, step=config.get('training.epochs'))
            # don't run eval again
            run_eval = False

    # Evaluation phase
    if run_eval:
        model_path = config.get('model.saved_path')
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state' in checkpoint:
                    model.load_state_dict(checkpoint['model_state'], strict=False)
                    print(f"Loaded trained model from {model_path}")
                else:
                    print(f"Warning: Invalid checkpoint format in {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
        
        model.to(device)
        evaluate_model(model, test_loader, device, config, logger, step=config.get('training.epochs'))
    
    logger.close()
    
if __name__ == "__main__":
    main()
