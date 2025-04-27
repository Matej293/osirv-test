import os 
import shutil
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from pathlib import Path

from metrics.base_logger import BaseLogger


class WandbLogger(BaseLogger):
    """Weights & Biases implementation of the BaseLogger, now with segmentation eval support."""
    
    def __init__(self, project=None, name=None, config=None, reuse=False):
        super().__init__()
        self.project = project
        self.name = name
        self.config = config
        self.train_batch_step = 0
        self.train_epoch_step = 1
        self.eval_step = 1

        # force wandb to write into repo/wandb
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wandb_dir = os.path.join(project_root, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        os.environ["WANDB_DIR"] = wandb_dir
        
        if not reuse and wandb.run is None:
            try:
                wandb.init(project=project, name=name, config=config)
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")

        self._define_metric_groups()
    
    
    def _define_metric_groups(self):
        wandb.define_metric("train_batch_step", summary="max")
        wandb.define_metric("train_epoch_step", summary="max")
        wandb.define_metric("eval_step", summary="max")
        
        wandb.define_metric("Train/Batch*", step_metric="train_batch_step")
        
        wandb.define_metric("Train/Epoch*",      step_metric="train_epoch_step")
        wandb.define_metric("Train/Accuracy",    step_metric="train_epoch_step")
        wandb.define_metric("Train/LearningRate",step_metric="train_epoch_step")
        wandb.define_metric("Train/WeightDecay", step_metric="train_epoch_step")

        wandb.define_metric("Eval/*", step_metric="eval_step")
    
    
    def _update_step_counter(self, tag, log_dict, step=None):
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
        if value is None:
            return
        log_dict = {tag: value}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    
    def log_histogram(self, tag, values, step=None, n_bins=20):
        """Log a distribution as a wandb.Histogram."""
        if values is None or len(values) == 0:
            return
        hist = wandb.Histogram(np_histogram=np.histogram(values, bins=n_bins))
        log_dict = {tag: hist}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    
    def log_images(self, tag, images, step=None, max_images=4):
        if images is None or len(images) == 0:
            return
        images_to_log = []
        for i in range(min(len(images), max_images)):
            img = images[i]
            if isinstance(img, torch.Tensor):
                arr = img.cpu().numpy()
                if arr.ndim == 3 and arr.shape[0] == 3:
                    arr = np.transpose(arr, (1, 2, 0))
                images_to_log.append(wandb.Image(arr))
            else:
                images_to_log.append(wandb.Image(img))
        log_dict = {tag: images_to_log}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
    
    
    def _log_figure(self, tag, figure, step=None):
        log_dict = {tag: wandb.Image(figure)}
        log_dict = self._update_step_counter(tag, log_dict, step)
        wandb.log(log_dict)
        plt.close(figure)
    
    def log_confusion_matrix(self, preds, targets, class_names, step=None, tag="ConfusionMatrix"):
        try:
            fig = self.create_confusion_matrix_figure(preds, targets, class_names)
            tag = f"Eval/{tag}"
            self._log_figure(tag, fig, step)
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    

    def log_evaluation_metrics(self, all_preds, all_targets, accuracy=None, avg_loss=None, step=None, 
                            class_names=["HP","SSA"], tag_prefix="Eval"):
        """Log evaluation metrics."""
        try:
            # ensure numpy ints
            preds_np   = self._ensure_numpy_int(all_preds).ravel()
            targets_np = self._ensure_numpy_int(all_targets).ravel()

            # compute scalar metrics
            metrics = self.calculate_metrics(preds_np, targets_np)
            if accuracy is not None:
                metrics['accuracy'] = accuracy
            if avg_loss is not None:
                metrics['loss'] = avg_loss

            log_dict = {}
            # log scalars
            for metric_name, value in metrics.items():
                tag = f"{tag_prefix}/{metric_name.capitalize()}"
                log_dict[tag] = value

            # confusion matrix
            # note: log_confusion_matrix will call wandb.log internally
            self.log_confusion_matrix(preds_np, targets_np,
                                        class_names=class_names,
                                        step=step,
                                        tag=f"{tag_prefix}/ConfusionMatrix")

            # now build and log the predictions vs labels table
            rows = [{"label": int(t), "prediction": int(p)}
                    for p, t in zip(preds_np, targets_np)]
            table = wandb.Table(columns=["label","prediction"], data=rows)
            log_dict[f"{tag_prefix}/PredictionTable"] = table

            # dispatch a single wandb.log call
            log_dict = self._update_step_counter(f"{tag_prefix}/IoU", log_dict, step)
            wandb.log(log_dict)

            return metrics

        except Exception as e:
            print(f"Error in log_evaluation_metrics: {e}")
            return {}

    
    
    def log_model(self, model_path, name=None, metadata=None):
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            return
        
        model_artifact = wandb.Artifact(
            name or "mhist-model",
            type="model",
            description="DeepLabV3+ model for MHIST segmentation",
            metadata=metadata
        )
        model_artifact.add_file(model_path)
        try:
            art_ref = wandb.log_artifact(model_artifact)
            art_ref.wait()
            wandb.log({})
            print(f"Model uploaded as artifact: {art_ref.name}")
        except Exception as e:
            print(f"Error uploading model artifact: {e}")
            return
        
        # clean up cache
        try:
            cache_dir = os.environ.get("WANDB_CACHE_DIR",
                                       os.path.join(Path.home(), ".cache", "wandb"))
            artifacts_dir = os.path.join(cache_dir, "artifacts")
            if os.path.exists(artifacts_dir):
                shutil.rmtree(artifacts_dir)
                os.makedirs(artifacts_dir)
            print("Cleaned wandb artifacts cache")
        except Exception as e:
            print(f"Warning: Could not clean cache: {e}")
    
    
    def close(self):
        if wandb.run is not None:
            wandb.finish()
