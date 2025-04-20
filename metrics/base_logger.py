import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score
from abc import ABC, abstractmethod

class BaseLogger(ABC):
    """Base logger class with common functionality for metrics calculation."""

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
    
    def calculate_metrics(self, preds, targets, average="binary"):
        """
        Calculate classification metrics.
        
        Args:
            preds: Predicted labels
            targets: Ground truth labels
            average: Averaging strategy for sklearn metrics
            
        Returns:
            Dictionary with precision, recall, f1, f2, accuracy, and iou metrics
        """
        preds_np = self._ensure_numpy_int(preds)
        targets_np = self._ensure_numpy_int(targets)
        
        # calculate IoU
        intersection = np.sum(preds_np * targets_np)
        union = np.sum(preds_np) + np.sum(targets_np) - intersection
        iou = intersection / (union + 1e-6)
        
        return {
            'precision': precision_score(targets_np, preds_np, zero_division=0, average=average),
            'recall': recall_score(targets_np, preds_np, zero_division=0, average=average),
            'f1': f1_score(targets_np, preds_np, zero_division=0, average=average),
            'f2': fbeta_score(targets_np, preds_np, beta=2, zero_division=0, average=average),
            'accuracy': np.mean(preds_np == targets_np),
            'iou': iou
        }
    
    def create_confusion_matrix_figure(self, preds, targets, class_names):
        """Create a confusion matrix figure."""
        preds_np = self._ensure_numpy_int(preds)
        targets_np = self._ensure_numpy_int(targets)
        
        cm = confusion_matrix(targets_np, preds_np)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        
        return fig
    
    @abstractmethod
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value."""
        pass
    
    @abstractmethod
    def log_images(self, tag, images, step=None, max_images=4):
        """Log images."""
        pass
    
    @abstractmethod
    def _log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure."""
        pass
    
    @abstractmethod
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        """Create and log a confusion matrix."""
        pass
        
    @abstractmethod
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth."""
        pass
    
    @abstractmethod
    def log_evaluation_metrics(self, all_preds, all_targets, accuracy=None, avg_loss=None, step=None, 
                              class_names=None, tag_prefix="Eval"):
        """Log evaluation metrics."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the logger."""
        pass