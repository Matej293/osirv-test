import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

from metrics.base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project=None, name=None, config=None):
        super().__init__()
        self.project = project
        self.name = name
        self.config = config
        
        self.train_batch_step = 0
        self.train_epoch_step = 0
        self.eval_step = 0
        
        if wandb.run is None:
            wandb.init(project=project, name=name, config=config)
        
        self._define_metric_groups()
    
    def _define_metric_groups(self):
        """Define metric groups with their own step counters."""
        wandb.define_metric("train_batch_step", summary="max")
        wandb.define_metric("train_epoch_step", summary="max")
        wandb.define_metric("eval_step", summary="max")
        
        wandb.define_metric("Train/Batch*", step_metric="train_batch_step")
        
        wandb.define_metric("Train/Epoch*", step_metric="train_epoch_step")
        wandb.define_metric("Train/Accuracy", step_metric="train_epoch_step")
        wandb.define_metric("Train/LearningRate", step_metric="train_epoch_step")
        
        wandb.define_metric("Eval/*", step_metric="eval_step")
    
    def log_scalar(self, tag, value, step=None):
        """Log a scalar value with the appropriate step counter."""
        log_dict = {tag: value}
        
        if tag.startswith("Train/Batch"):
            if step is not None:
                self.train_batch_step = max(self.train_batch_step, step)
            else:
                self.train_batch_step += 1
            
            log_dict["train_batch_step"] = self.train_batch_step
        
        elif tag.startswith("Train/"):
            if step is not None:
                self.train_epoch_step = max(self.train_epoch_step, step)
            else:
                self.train_epoch_step += 1
                
            log_dict["train_epoch_step"] = self.train_epoch_step
        
        elif tag.startswith("Eval/"):
            if step is not None:
                self.eval_step = max(self.eval_step, step)
            else:
                self.eval_step += 1
                
            log_dict["eval_step"] = self.eval_step
        
        wandb.log(log_dict)
    
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
            
            self.log_scalar("Eval/Precision", precision, step)
            self.log_scalar("Eval/Recall", recall, step)
            self.log_scalar("Eval/F1-Score", f1, step)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def log_evaluation(self, all_preds, all_targets, accuracy, avg_loss, step=None):
        """Log full evaluation results."""
        dice = self.calculate_dice_coefficient(all_preds, all_targets)
        iou = self.calculate_iou(all_preds, all_targets)
        class_metrics = self.calculate_per_class_dice_iou(all_preds, all_targets)
        
        self.log_scalar("Eval/Accuracy", accuracy, step)
        self.log_scalar("Eval/Loss", avg_loss, step)
        self.log_scalar("Eval/Dice", dice, step)
        self.log_scalar("Eval/IoU", iou, step)
        self.log_scalar("Eval/Dice_HP", class_metrics['dice_hp'], step)
        self.log_scalar("Eval/Dice_SSA", class_metrics['dice_ssa'], step)
        self.log_scalar("Eval/IoU_HP", class_metrics['iou_hp'], step)
        self.log_scalar("Eval/IoU_SSA", class_metrics['iou_ssa'], step)
        
        self.log_metrics(all_preds, all_targets, step)
        
        if step is not None:
            self.eval_step = max(self.eval_step, step)
        else:
            self.eval_step += 1
            
        self.log_confusion_matrix(
            all_preds, all_targets,
            class_names=["HP", "SSA"], 
            step=self.eval_step
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