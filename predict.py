import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from network import modeling
from metrics.logger import Logger
from datasets.mhist import get_mhist_dataloader

# default configuration
MODEL_PATH = "models/deeplabv3plus_resnet101.pth"
SAVE_MODEL_PATH = "models/deeplabv3plus_resnet101_trained.pth"
CSV_PATH = "data/mhist_annotations.csv"
IMG_DIR = "data/images"
BATCH_SIZE = 16
EPOCHS = 10
L_R = 3e-5
WEIGHT_DECAY = 5e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssa_count = 990
hp_count = 2162
total_count = ssa_count + hp_count
CLASS_WEIGHTS = torch.tensor([total_count / hp_count, total_count / ssa_count], device=device)

# parsing command-line arguments
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval"],
                        help="Mode: train or eval")
    parser.add_argument("--csv_path", type=str, default=CSV_PATH,
                        help="Path to annotations CSV file")
    parser.add_argument("--img_dir", type=str, default=IMG_DIR,
                        help="Path to image directory")
    parser.add_argument("--model_path", type=str, default=MODEL_PATH,
                        help="Path to pre-trained model")
    parser.add_argument("--save_model_path", type=str, default=SAVE_MODEL_PATH,
                        help="Path to save trained model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training/testing")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=L_R,
                      help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY,
                      help="Weight decay for optimizer")
    parser.add_argument("--gpu_id", type=str, default="0",
                        help="GPU ID to use for training/testing")
    return parser

# Training function
def train_model(model, train_loader, device, epochs, logger, save_path, lr=L_R, weight_decay=WEIGHT_DECAY):
    criterion = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.3, patience=2, threshold=0.01, min_lr=1e-6
    )
    
    global_step = 0 # used by tensorboard logger
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        correct, total = 0, 0

        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # batch metrics
            batch_loss = loss.item()
            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            batch_correct = (predicted == masks).sum().item()
            batch_total = masks.numel()
            batch_accuracy = batch_correct / batch_total

            # Update epoch metrics
            total_loss += batch_loss
            correct += batch_correct
            total += batch_total

            # batch-level metrics
            if batch_idx % 100 == 0:
                logger.log_scalar("Train/BatchLoss", batch_loss, global_step)
                logger.log_scalar("Train/BatchAccuracy", batch_accuracy, global_step)

            # logging learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_scalar("Train/LearningRate", current_lr, global_step)

            global_step += 1

        # epoch metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # epoch-level metrics
        logger.log_scalar("Train/EpochLoss", avg_loss, epoch)
        logger.log_scalar("Train/EpochAccuracy", accuracy, epoch)

        scheduler.step(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}")

    torch.save({'model_state': model.state_dict()}, save_path)
    print(f"Model saved to {save_path}")

# Evaluation function
def evaluate_model(model, test_loader, device, logger):
    model.eval()
    criterion = nn.BCEWithLogitsLoss(pos_weight=CLASS_WEIGHTS[1])

    total_loss = 0.0
    correct, total = 0, 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            masks = masks.unsqueeze(1).float()

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()

            correct += (predicted == masks).sum().item()
            total += masks.numel()
           
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(masks.cpu().numpy().flatten())

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    logger.log_evaluation(all_preds, all_targets, accuracy, avg_loss)

    print(f"Evaluation — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# Main function
def main():
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and opts.gpu_id != "-1" else "cpu")
    print(f"Using device: {device}")

    logger = Logger(log_dir="runs/mhist_experiment")

    # Load Dataset
    train_loader = get_mhist_dataloader(opts.csv_path, opts.img_dir, opts.batch_size, partition="train")
    test_loader = get_mhist_dataloader(opts.csv_path, opts.img_dir, opts.batch_size, partition="test")

    # Load model
    NUM_CLASSES = 1
    model = modeling.deeplabv3plus_resnet101(num_classes=NUM_CLASSES, output_stride=16, pretrained_backbone=True)

    # Modify classifier for binary classification
    model.classifier = modeling.DeepLabHeadV3Plus(
        in_channels=2048, low_level_channels=256, num_classes=NUM_CLASSES, aspp_dilate=[12, 24, 36]
    )

    if opts.mode == "train":
        if os.path.exists(opts.model_path):
            checkpoint = torch.load(opts.model_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state']
            del state_dict['classifier.classifier.3.weight']
            del state_dict['classifier.classifier.3.bias']
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded model from {opts.model_path}")
        
        model.to(device)
        train_model(
            model, 
            train_loader, 
            device, 
            opts.epochs, 
            logger,
            opts.save_model_path,
            lr=opts.lr,
            weight_decay=opts.weight_decay
        )

    elif opts.mode == "eval":
        if os.path.exists(opts.save_model_path):
            checkpoint = torch.load(opts.save_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state'])
            print(f"Loaded trained model from {opts.save_model_path}")
        else:
            print(f"Error: Model file not found at {opts.save_model_path}")
            return
        
        model.to(device)
        evaluate_model(model, test_loader, device, logger)
    
    logger.close()
    
if __name__ == "__main__":
    main()
