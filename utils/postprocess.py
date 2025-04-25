import torch
import numpy as np
from skimage import morphology
from scipy import ndimage

def post_process_segmentation(predictions, min_size=50, closing_radius=3, fill_holes=True):
    """
    Apply post-processing techniques to improve segmentation quality.
    
    Args:
        predictions: Tensor of shape (B, 1, H, W) with sigmoid outputs
        min_size: Minimum connected component size to keep
        closing_radius: Radius for morphological closing operation
        fill_holes: Whether to fill holes in the mask
        
    Returns:
        Processed predictions as tensor with same shape
    """
    processed_batch = []
    
    for pred in predictions:
        mask = pred.cpu().numpy().squeeze()
        
        binary_mask = mask > 0.5
        
        cleaned = morphology.remove_small_objects(binary_mask, min_size=min_size)
        
        if closing_radius > 0:
            closed = morphology.closing(cleaned, morphology.disk(closing_radius))
        else:
            closed = cleaned
            
        if fill_holes:
            filled = ndimage.binary_fill_holes(closed)
        else:
            filled = closed

        processed = torch.from_numpy(filled.astype(np.float32)).to(pred.device)
        processed_batch.append(processed.unsqueeze(0))
    
    return torch.cat(processed_batch, dim=0)