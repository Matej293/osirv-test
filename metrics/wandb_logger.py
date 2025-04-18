import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from metrics.base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project="mhist-classification", name=None, config=None, init_timeout=180):
        """Initialize wandb logger."""
        super().__init__()
        
        if wandb.run is not None:
            self.run = wandb.run
            if config:
                for key, value in config.items():
                    if key not in self.run.config:
                        self.run.config[key] = value
            
            self._define_metric_groups()
            return

        settings = wandb.Settings(init_timeout=init_timeout)
        self.run = wandb.init(
            project=project, 
            name=name, 
            config=config,
            settings=settings,
            reinit="return_previous",
            job_type="train"
        )
        
        self._define_metric_groups()

    def _define_metric_groups(self):
        """Define metric groups with their own step counters."""
        # batch metrics
        wandb.define_metric("Train/BatchLoss", step_metric="batch_step")
        wandb.define_metric("Train/BatchAccuracy", step_metric="batch_step")
        
        # epoch metrics
        wandb.define_metric("Train/EpochLoss", step_metric="epoch_step")
        wandb.define_metric("Train/Accuracy", step_metric="epoch_step")
        wandb.define_metric("Train/LearningRate", step_metric="epoch_step")
        
        # eval metrics
        wandb.define_metric("Eval/*", step_metric="epoch_step")
    
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value to wandb using the appropriate step counter."""
        if step is None:
            wandb.log({tag: value})
            return
        
        if tag.startswith("Train/Batch"):
            wandb.log({tag: value, "batch_step": step})
        elif tag.startswith(("Train/Epoch", "Train/Accuracy", "Train/Learning", "Eval/")):
            wandb.log({tag: value, "epoch_step": step})
        else:
            wandb.log({tag: value})
    
    def log_text(self, text, step=None):
        """Log text to W&B."""
        wandb.log({"Text": wandb.Html(text)}, step=step)

    def log_images(self, tag, images, step=None, max_images=4):
        """Log images to wandb."""
        if images is None or len(images) == 0:
            print(f"Warning: No images to log for {tag}")
            return
        
        images_to_log = []
        for i in range(min(len(images), max_images)):
            if isinstance(images[i], torch.Tensor):
                img = images[i].cpu().numpy()
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                images_to_log.append(wandb.Image(img))
            else:
                images_to_log.append(wandb.Image(images[i]))
                
        wandb.log({tag: images_to_log}, step=step)
    
    def _log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure to W&B."""
        wandb.log({tag: wandb.Image(figure)}, step=step)
    
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth."""
        self.log_images(f"{tag_prefix}/Input", images, step)
        self.log_images(f"{tag_prefix}/Prediction", preds, step)
        self.log_images(f"{tag_prefix}/GroundTruth", targets, step)
    
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        """Create and log a confusion matrix."""
        try:
            fig = self.create_confusion_matrix_figure(preds, targets, class_names)
            self._log_figure(tag, fig, step)
            plt.close(fig)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def log_metrics(self, preds, targets, step, average="binary"):
        """Calculate and log classification metrics."""
        try:
            preds_np = self._ensure_numpy_int(preds)
            targets_np = self._ensure_numpy_int(targets)
            
            precision = precision_score(targets_np, preds_np, zero_division=0, average=average)
            recall = recall_score(targets_np, preds_np, zero_division=0, average=average)
            f1 = f1_score(targets_np, preds_np, zero_division=0, average=average)
            
            wandb.log({
                "Eval/Precision": precision,
                "Eval/Recall": recall,
                "Eval/F1-Score": f1
            }, step=step)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def log_evaluation(self, all_preds, all_targets, accuracy, avg_loss, step=None):
        """Log full evaluation results."""
        dice = self.calculate_dice_coefficient(all_preds, all_targets)
        iou = self.calculate_iou(all_preds, all_targets)
        class_metrics = self.calculate_per_class_dice_iou(all_preds, all_targets)
        
        wandb.log({
            "Eval/Accuracy": accuracy,
            "Eval/Loss": avg_loss,
            "Eval/Dice": dice,
            "Eval/IoU": iou,
            "Eval/Dice_HP": class_metrics['dice_hp'],
            "Eval/Dice_SSA": class_metrics['dice_ssa'],
            "Eval/IoU_HP": class_metrics['iou_hp'],
            "Eval/IoU_SSA": class_metrics['iou_ssa']
        }, step=step)
        
        self.log_metrics(all_preds, all_targets, step)
        
        self.log_confusion_matrix(
            all_preds, all_targets,
            class_names=["HP", "SSA"], 
            step=step
        )
        
        return dice, iou
    
    def log_model(self, model_path, name=None, metadata=None):
        """Log model as a wandb artifact."""
        model_artifact = wandb.Artifact(
            name or "mhist-model", 
            type="model",
            description="DeepLabV3+ model for MHIST classification",
            metadata=metadata
        )
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
    
    def close(self):
        """Finish the wandb run."""
        wandb.finish()