import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

class BaseLogger:
    """Base logger class with common functionality for all loggers."""

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
    
    def calculate_f1_coefficient(self, y_pred, y_true):
        """
        Calculate Dice coefficient
        Formula: 2*|X∩Y|/(|X|+|Y|)
        """
        y_pred = self._ensure_numpy_int(y_pred)
        y_true = self._ensure_numpy_int(y_true)

        return f1_score(y_true, y_pred, zero_division=0, average='binary')

    def calculate_iou(self, y_pred, y_true):
        """
        Calculate IoU (Jaccard Index)
        Formula: |X∩Y|/|X∪Y|
        """
        y_pred = self._ensure_numpy_int(y_pred)
        y_true = self._ensure_numpy_int(y_true)
        
        intersection = np.sum(y_pred * y_true)
        union = np.sum(y_pred) + np.sum(y_true) - intersection
        return intersection / (union + 1e-6)
    
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
    
    def calculate_metrics(self, preds, targets, average="binary"):
        """Calculate classification metrics."""
        preds_np = self._ensure_numpy_int(preds)
        targets_np = self._ensure_numpy_int(targets)
        
        precision = precision_score(targets_np, preds_np, zero_division=0, average=average)
        recall = recall_score(targets_np, preds_np, zero_division=0, average=average)
        f1 = f1_score(targets_np, preds_np, zero_division=0, average=average)
        f2 = fbeta_score(targets_np, preds_np, beta=2, zero_division=0, average=average)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f2': f2
        }
    
    def log_metrics(self, preds, targets, step, average="binary"):
        """Calculate and log classification metrics."""
        try:
            metrics = self.calculate_metrics(preds, targets, average)
            
            self.log_scalar("Eval/Precision", metrics['precision'], step)
            self.log_scalar("Eval/Recall", metrics['recall'], step)
            self.log_scalar("Eval/F1-Score", metrics['f1'], step)
            self.log_scalar("Eval/F2-Score", metrics['f2'], step)

            return metrics
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {}
    
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        """Create and log a confusion matrix."""
        try:
            fig = self.create_confusion_matrix_figure(preds, targets, class_names)
            self._log_figure(tag, fig, step)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def log_evaluation(self, all_preds, all_targets, accuracy, avg_loss, step=None):
        """Log full evaluation results."""
        dice = self.calculate_f1_coefficient(all_preds, all_targets)
        iou = self.calculate_iou(all_preds, all_targets)
        
        self.log_scalar("Eval/Accuracy", accuracy, step)
        self.log_scalar("Eval/Loss", avg_loss, step)
        self.log_scalar("Eval/IoU", iou, step)
        
        self.log_metrics(all_preds, all_targets, step)
        
        self.log_confusion_matrix(
            all_preds, all_targets,
            class_names=["HP", "SSA"], 
            step=step
        )
        
        return dice, iou
    
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth."""
        self.log_images(f"{tag_prefix}/Input", images, step)
        self.log_images(f"{tag_prefix}/Prediction", preds, step)
        self.log_images(f"{tag_prefix}/GroundTruth", targets, step)
    
    # to be implemented by subclasses
    def log_scalar(self, tag, value, step):
        raise NotImplementedError
    
    def log_images(self, tag, images, step, max_images=4):
        raise NotImplementedError
    
    def _log_figure(self, tag, figure, step=None):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError