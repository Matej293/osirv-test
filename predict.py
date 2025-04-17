import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from network import modeling
from metrics.tensorboard_logger import TensorboardLogger
from metrics.wandb_logger import WandbLogger
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from utils.visualization import visualize_segmentation_results

# parsing command-line arguments
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default_config.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"],
                        help="Mode: train or eval")
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
    parser.add_argument("--ssa_threshold", type=float,
                      help="Classification threshold for SSA class")
    parser.add_argument("--hp_threshold", type=float,
                      help="Classification threshold for HP class")
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="mhist-classification",
                      help="WandB project name")
    parser.add_argument("--wandb_name", type=str, default=None,
                      help="WandB run name")
    parser.add_argument("--init_timeout", type=int, default=180,
                      help="Timeout for wandb initialization")
    return parser

# Training function
def train_model(model, train_loader, device, config, logger=None, save_path=None, distributed=False, train_sampler=None):
    """Train the model with the given configuration."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
    
    lr = float(config.get('training.learning_rate'))
    weight_decay = float(config.get('training.weight_decay'))
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config.get('training.scheduler.mode', 'max'),
        factor=config.get('training.scheduler.factor', 0.2),
        patience=config.get('training.scheduler.patience', 2),
        threshold=config.get('training.scheduler.threshold', 0.005),
        min_lr=config.get('training.scheduler.min_lr', 1e-6)
    )
    
    # Training loop
    global_step = 0
    model.train()
    epochs = config.get('training.epochs')
    log_interval = config.get('logging.batch_log_interval')
    ssa_threshold = config.get('training.ssa_threshold')
    hp_threshold = config.get('training.hp_threshold')
    
    best_metric = 0.0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct, total = 0, 0
        
        # Set epoch for distributed sampler if using distributed training
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

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
            predicted = torch.zeros_like(masks)
            predicted[probs > ssa_threshold] = 1.0
            predicted[probs < hp_threshold] = 0.0
            
            # updating batch metrics
            batch_correct = (predicted == masks).sum().item()
            batch_total = masks.numel()
            
            epoch_loss += batch_loss
            correct += batch_correct
            total += batch_total
            
            global_step += 1
            
            # logging batch results
            if logger and batch_idx % log_interval == 0:
                batch_accuracy = batch_correct / batch_total
                
                # Use log_scalar instead of log_metrics
                logger.log_scalar('Train/BatchLoss', batch_loss, step=global_step)
                logger.log_scalar('Train/BatchAccuracy', batch_accuracy, step=global_step)
                logger.log_scalar('Train/LearningRate', optimizer.param_groups[0]['lr'], step=global_step)

        # epoch metrics
        epoch_loss /= len(train_loader)
        accuracy = correct / total

        logger.log_scalar('Train/EpochLoss', epoch_loss, step=global_step)
        logger.log_scalar('Train/Accuracy', accuracy, step=global_step)
        
        scheduler.step(accuracy)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, "
              f"Accuracy: {accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model - handle DDP case
        if save_path and accuracy > best_metric:
            best_metric = accuracy

            if distributed and hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            
            if logger:
                logger.log_text(f"Saved best model with {config.get('training.scheduler.mode')}={accuracy:.4f}")

    return model

# Evaluation function
def evaluate_model(model, test_loader, device, config, logger, step=None):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=config.class_weights[1])
    
    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)
            
            # applying class-specific thresholds
            predicted = torch.zeros_like(masks)
            predicted[probs > config.get('training.ssa_threshold')] = 1.0
            predicted[probs < config.get('training.hp_threshold')] = 0.0
            
            correct += (predicted == masks).sum().item()
            total += masks.numel()
           
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    # logging evaluation metrics
    dice, iou = logger.log_evaluation(all_preds, all_targets, accuracy, avg_loss, step=step)
    class_metrics = logger.calculate_per_class_dice_iou(all_preds, all_targets)

    print(f"Evaluation — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Dice Coefficient: {dice:.4f}, IoU: {iou:.4f}")
    print(f"HP - Dice: {class_metrics['dice_hp']:.4f}, IoU: {class_metrics['iou_hp']:.4f}")
    print(f"SSA - Dice: {class_metrics['dice_ssa']:.4f}, IoU: {class_metrics['iou_ssa']:.4f}")

    # logging sample images
    if len(test_loader) > 0:
        try:
            sample_images, sample_masks = next(iter(test_loader))
            sample_images = sample_images.to(device)
            sample_masks = sample_masks.unsqueeze(1).float().to(device)
            
            with torch.no_grad():
                sample_outputs = model(sample_images)
                sample_probs = torch.sigmoid(sample_outputs)
                
                sample_preds = torch.zeros_like(sample_masks)
                sample_preds[sample_probs > config.get('training.ssa_threshold')] = 1.0
                sample_preds[sample_probs < config.get('training.hp_threshold')] = 0.0
            
            # visualizing the data as images
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

    # logging to the wandb site
    if isinstance(logger, WandbLogger):
        metadata = {
            "accuracy": accuracy,
            "dice": dice,
            "iou": iou,
            "dice_ssa": class_metrics['dice_ssa'],
            "hp_threshold": config.get("training.hp_threshold"),
            "ssa_threshold": config.get("training.ssa_threshold")
        }
        logger.log_model(
            config.get('model.saved_path'),
            name=f"mhist-model-acc{accuracy:.4f}-dice{dice:.4f}",
            metadata=metadata
        )

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

    if args.use_wandb:
        wandb_config = {
            "model": config.get("model"),
            "data": config.get("data"),
            "training": config.get("training"),
            "augmentation": config.get("augmentation"),
        }
        
        logger = WandbLogger(
            project=args.wandb_project,
            name=args.wandb_name,
            config=wandb_config,
            init_timeout=args.init_timeout
        )
        print("Using Weights & Biases for logging")
    else:
        logger = TensorboardLogger(log_dir=config.get('logging.log_dir'))
        print("Using TensorBoard for logging")

    # loader setup
    train_loader = get_mhist_dataloader(
        config.get('data.csv_path'), 
        config.get('data.img_dir'), 
        config.get('data.batch_size'),
        partition="train",
        augmentation_config=config.get('augmentation.train')
    )
    
    test_loader = get_mhist_dataloader(
        config.get('data.csv_path'), 
        config.get('data.img_dir'), 
        config.get('data.batch_size'),
        partition="test",
        augmentation_config=config.get('augmentation.test')
    )

    # model setup
    model = modeling.deeplabv3plus_resnet101(
        num_classes=config.get('model.num_classes'),
        output_stride=config.get('model.output_stride'),
        pretrained_backbone=config.get('model.pretrained_backbone')
    )

    model.classifier = modeling.DeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256, 
        num_classes=config.get('model.num_classes'),
        aspp_dilate=config.get('model.aspp_dilate')
    )

    if args.mode == "train":
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
        
        model.to(device)
        train_model(
            model, 
            train_loader, 
            device, 
            config,
            logger,
            config.get('model.saved_path')
        )

    elif args.mode == "eval":
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

        evaluate_model(model, test_loader, device, config, logger)
    
    logger.close()
    
if __name__ == "__main__":
    main()
