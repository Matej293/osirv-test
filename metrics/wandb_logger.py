import os
import shutil
import torch
import numpy as np
import wandb
from pathlib import Path
import matplotlib.pyplot as plt

from metrics.base_logger import BaseLogger

class WandbLogger(BaseLogger):
    """Weights & Biases implementation of the BaseLogger."""
    
    def __init__(self, project=None, name=None, config=None, reuse=False):
        """Initialize the WandB logger.
        
        Args:
            project: WandB project name
            name: WandB run name
            config: Configuration to log
            reuse: Whether to reuse an existing run
        """
        super().__init__()
        self.project = project
        self.name = name
        self.config = config
        self.train_batch_step = 0
        self.train_epoch_step = 1
        self.eval_step = 1

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wandb_dir = os.path.join(project_root, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        
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
        wandb.define_metric("Train/WeightDecay", step_metric="train_epoch_step")

        wandb.define_metric("Eval/*", step_metric="eval_step")
        
    def _update_step_counter(self, tag, log_dict, step=None):
        """Update the appropriate step counter based on the tag prefix.
        
        Args:
            tag: Metric tag
            log_dict: Dictionary to log
            step: Optional step value
            
        Returns:
            Updated log_dict with step counter
        """
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
        """Log a scalar value.
        
        Args:
            tag: Metric name
            value: Value to log
            step: Optional step value
        """
        if value is None:
            return
            
        log_dict = {tag: value}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
        
    def log_images(self, tag, images, step=None, max_images=4):
        """Log images to wandb.
        
        Args:
            tag: Image group name
            images: List of images (tensors or numpy arrays)
            step: Optional step value
            max_images: Maximum number of images to log
        """
        if images is None or len(images) == 0:
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
        """Log a matplotlib figure.
        
        Args:
            tag: Figure name
            figure: Matplotlib figure
            step: Optional step value
        """
        log_dict = {tag: wandb.Image(figure)}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
        plt.close(figure)
    
    def log_predictions(self, images, preds, targets, step, tag_prefix="Eval"):
        """Log input images, predictions, and ground truth.
        
        Args:
            images: Input images
            preds: Prediction masks/labels
            targets: Ground truth masks/labels
            step: Step value
            tag_prefix: Prefix for the tag
        """
        self.log_images(f"{tag_prefix}/Input", images, step)
        self.log_images(f"{tag_prefix}/Prediction", preds, step)
        self.log_images(f"{tag_prefix}/GroundTruth", targets, step)
        
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        """Create and log a confusion matrix.
        
        Args:
            preds: Predicted labels
            targets: Ground truth labels
            class_names: List of class names
            step: Optional step value
            tag: Tag for the confusion matrix
        """
        try:
            fig = self.create_confusion_matrix_figure(preds, targets, class_names)
            if tag == "ConfusionMatrix":
                tag = f"Eval/{tag}"
            self._log_figure(tag, fig, step)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            
    def log_evaluation_metrics(self, all_preds, all_targets, accuracy=None, avg_loss=None, step=None, 
                              class_names=["HP", "SSA"], tag_prefix="Eval"):
        """Log evaluation metrics.
        
        Args:
            all_preds: All predictions
            all_targets: All targets
            accuracy: Optional precalculated accuracy
            avg_loss: Optional average loss
            step: Optional step value
            class_names: List of class names
            tag_prefix: Prefix for metric tags
            
        Returns:
            Dictionary of metrics
        """
        try:
            preds_np = self._ensure_numpy_int(all_preds)
            targets_np = self._ensure_numpy_int(all_targets)

            metrics = self.calculate_metrics(preds_np, targets_np)
            
            if accuracy is not None:
                metrics['accuracy'] = accuracy
            if avg_loss is not None:
                metrics['loss'] = avg_loss
            
            for metric_name, value in metrics.items():
                if value is not None:
                    metric_tag = f"{tag_prefix}/{metric_name.capitalize()}"
                    self.log_scalar(metric_tag, value, step)
            
            self.log_confusion_matrix(preds_np, targets_np, class_names=class_names, step=step)
            
            return metrics
            
        except Exception as e:
            print(f"Error in log_evaluation_metrics: {e}")
            return {}
            
    def log_model(self, model_path, name=None, metadata=None):
        """Log model as a wandb artifact and clean up after uploading.
        
        Args:
            model_path: Path to the model file
            name: Optional name for the artifact
            metadata: Optional metadata dictionary
        """
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
            
        model_artifact = wandb.Artifact(
            name or "mhist-model", 
            type="model",
            description="DeepLabV3+ model for MHIST classification",
            metadata=metadata
        )
        model_artifact.add_file(model_path)
        
        try:
            artifact_ref = wandb.log_artifact(model_artifact)
            artifact_ref.wait()
            
            wandb.log({})
            print(f"Model uploaded as artifact: {artifact_ref.name}")
        except Exception as e:
            print(f"Error uploading model artifact: {e}")
            return

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
        if wandb.run is not None:
            wandb.finish()