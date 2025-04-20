import os
import shutil
import torch
import numpy as np
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

from metrics.base_logger import BaseLogger

class WandbLogger(BaseLogger):
    def __init__(self, project=None, name=None, config=None):
        super().__init__()
        self.project = project
        self.name = name
        self.config = config

        # set up wandb directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wandb_dir = os.path.join(project_root, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        
        self.train_batch_step = 0
        self.train_epoch_step = 1
        self.eval_step = 1
        
        try:
            wandb.init(project=project, name=name, config=config)
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
        
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
    
    def _update_step_counter(self, tag, log_dict, step=None):
        """Update the appropriate step counter based on the tag prefix and add it to log_dict."""
        if tag.startswith("Eval/"):
            if step is not None:
                self.eval_step = max(self.eval_step, step)
            log_dict["eval_step"] = self.eval_step
        elif tag.startswith("Train/Batch"):
            if step is not None:
                self.train_batch_step = max(self.train_batch_step, step)
            else:
                self.train_batch_step += 1
            log_dict["train_batch_step"] = self.train_batch_step
        else:
            if step is not None:
                self.train_epoch_step = max(self.train_epoch_step, step)
            else:
                self.train_epoch_step += 1
            log_dict["train_epoch_step"] = self.train_epoch_step
        
        return log_dict

    def log_scalar(self, tag, value, step=None):
        """Log a scalar value with the appropriate step counter."""
        log_dict = {tag: value}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    def log_images(self, tag, images, step=None, max_images=4):
        """Log images to wandb with appropriate step counter."""
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
        
        log_dict = {tag: images_to_log}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    def _log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure to W&B with appropriate step counter."""
        log_dict = {tag: wandb.Image(figure)}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth."""
        self.log_images(f"{tag_prefix}/Input", images, step)
        self.log_images(f"{tag_prefix}/Prediction", preds, step)
        self.log_images(f"{tag_prefix}/GroundTruth", targets, step)
    
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        """Create and log a confusion matrix."""
        try:
            fig = self.create_confusion_matrix_figure(preds, targets, class_names)
            if tag == "ConfusionMatrix":
                tag = f"Eval/{tag}"
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
            f2 = fbeta_score(targets_np, preds_np, beta=2, zero_division=0, average=average)
            
            self.log_scalar("Eval/Precision", precision, step)
            self.log_scalar("Eval/F1-Score", f1, step)
            self.log_scalar("Eval/F2-Score", f2, step)
            self.log_scalar("Eval/Recall", recall, step)
        except Exception as e:
            print(f"Error calculating metrics: {e}")
    
    def log_evaluation(self, all_preds, all_targets, accuracy, avg_loss, step=None):
        """Log full evaluation results."""
        f1 = self.calculate_f1_coefficient(all_preds, all_targets)
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
        
        return f1, iou
    
    def log_model(self, model_path, name=None, metadata=None):
        """Log model as a wandb artifact and clean up after uploading."""
        
        model_artifact = wandb.Artifact(
            name or "mhist-model", 
            type="model",
            description="DeepLabV3+ model for MHIST classification",
            metadata=metadata
        )
        model_artifact.add_file(model_path)
        artifact_ref = wandb.log_artifact(model_artifact)
        artifact_ref.wait()
        
        wandb.log({})
        print(f"Model uploaded as artifact: {artifact_ref.name}")

        # cleaning up cache
        try:
            cache_dir = os.environ.get("WANDB_CACHE_DIR", 
                                      os.path.join(Path.home(), ".cache", "wandb"))
            artifacts_dir = os.path.join(cache_dir, "artifacts")
            
            if os.path.exists(artifacts_dir):
                size_before = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for dirpath, dirnames, filenames in os.walk(artifacts_dir)
                    for f in filenames if os.path.exists(os.path.join(dirpath, f))
                ) / (1024 * 1024)  # MB
                
                print(f"Cleaning wandb artifacts cache (size: {size_before:.2f} MB)")
                shutil.rmtree(artifacts_dir)
                os.makedirs(artifacts_dir)
                
                print(f"Successfully cleaned wandb artifacts cache")
        except Exception as e:
            print(f"Warning: Could not clean wandb artifacts cache: {e}")
    
    def close(self):
        """Finish the wandb run."""
        wandb.finish()