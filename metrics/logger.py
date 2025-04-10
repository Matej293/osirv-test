import os
import torch
import torchvision
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

class Logger:
    def __init__(self, log_dir="runs/experiment"):
        """Initialize logger with TensorBoard writer."""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
    
    def log_scalar(self, tag, value, step):
        """Log a scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, step)
    
    def log_images(self, tag, images, step, max_images=4):
        """Log images to TensorBoard."""
        if images is None or len(images) == 0:
            print(f"Warning: No images to log for {tag}")
            return
        
        grid = torchvision.utils.make_grid(
            images[:max_images].cpu(), 
            normalize=True, 
            scale_each=True
        )
        self.writer.add_image(tag, grid, step)
    
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth."""
        self.log_images(f"{tag_prefix}/Input", images, step)
        self.log_images(f"{tag_prefix}/Prediction", preds, step)
        self.log_images(f"{tag_prefix}/GroundTruth", targets, step)
    
    def _ensure_numpy_int(self, data):
        """Convert data to numpy int array regardless of input type."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy().astype(int)
        elif isinstance(data, list):
            return np.array(data, dtype=int)
        elif isinstance(data, np.ndarray):
            return data.astype(int)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
    
    def log_confusion_matrix(self, preds, targets, class_names, step, tag="ConfusionMatrix"):
        """Create and log a confusion matrix."""
        try:
            # Convert inputs to numpy arrays
            preds_np = self._ensure_numpy_int(preds)
            targets_np = self._ensure_numpy_int(targets)
            
            # Create confusion matrix
            cm = confusion_matrix(targets_np, preds_np)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            
            self.writer.add_figure(tag, fig, step)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def log_metrics(self, preds, targets, step, average="binary"):
        """Calculate and log classification metrics."""
        try:
            # Convert inputs to numpy arrays
            preds_np = self._ensure_numpy_int(preds)
            targets_np = self._ensure_numpy_int(targets)
            
            # Calculate metrics
            precision = precision_score(targets_np, preds_np, zero_division=0, average=average)
            recall = recall_score(targets_np, preds_np, zero_division=0, average=average)
            f1 = f1_score(targets_np, preds_np, zero_division=0, average=average)
            
            # Log metrics
            self.log_scalar("Eval/Precision", precision, step)
            self.log_scalar("Eval/Recall", recall, step)
            self.log_scalar("Eval/F1-Score", f1, step)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def log_evaluation(self, all_preds, all_targets, accuracy, avg_loss, step=0):
        """Log full evaluation results."""
        # Log basic metrics
        self.log_scalar("Eval/Accuracy", accuracy, step)
        self.log_scalar("Eval/Loss", avg_loss, step)
        
        # Log classification metrics
        self.log_metrics(all_preds, all_targets, step)
        
        # Log confusion matrix
        self.log_confusion_matrix(
            all_preds, all_targets,
            class_names=["HP", "SSA"], 
            step=step
        )
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
