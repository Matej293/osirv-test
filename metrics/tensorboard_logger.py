import os
import torchvision
from torch.utils.tensorboard import SummaryWriter

from metrics.base_logger import BaseLogger

class TensorboardLogger(BaseLogger):
    def __init__(self, log_dir="runs/experiment"):
        """Initialize logger with TensorBoard writer."""
        super().__init__()
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
    
    def _log_figure(self, tag, figure, step=None):
        """Log a matplotlib figure to TensorBoard."""
        self.writer.add_figure(tag, figure, step)
    
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
