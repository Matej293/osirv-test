import numpy as np
import torch
import cv2
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchstain
import albumentations as A

# Torchstain: Macenko Normalizer
class TorchstainNormalize(A.ImageOnlyTransform):
    """Albumentations wrapper around torchstain.MacenkoNormalizer."""
    def __init__(self, always_apply=False, p=0.7):
        super().__init__(always_apply, p)
        macenko = torchstain.normalizers.MacenkoNormalizer()
        self.normalizer = macenko

    def apply(self, img, **kwargs):
        # img: H×W×C uint8 RGB
        # torchstain expects float [0,1]
        img_f = img.astype("float32") / 255.0
        out = self.normalizer.normalize(img_f)
        # back to uint8
        out = (np.clip(out, 0, 1) * 255).astype("uint8")
        return out

    def get_transform_init_args_names(self):
        return ()
    
# Loss: Lovasz hinge
def lovasz_grad(gt_sorted):
    """
    Compute gradient of the Lovasz extension w.r.t sorted errors
    gt_sorted: [P] ground truth labels (1 or 0) sorted by errors descending
    """
    p = gt_sorted.sum()
    n = gt_sorted.size(0) - p
    intersection = p - gt_sorted.float().cumsum(0)
    union = p + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if n > 0:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard

def flatten_binary_scores(scores, labels):
    """
    Flattens predictions in the batch
    scores: [B, H, W] logits
    labels: [B, H, W] ground truth binary masks
    Returns:
        scores_flat: [P] flattened logits
        labels_flat: [P] flattened labels
    """
    scores_flat = scores.view(-1)
    labels_flat = labels.view(-1)
    return scores_flat, labels_flat

def lovasz_hinge(logits, labels, per_image=True):
    """
    Binary Lovasz hinge loss
    logits: [B,1,H,W] logits from network
    labels: [B,1,H,W] binary ground truth (0 or 1)
    """
    if per_image:
        loss = []
        for log, lab in zip(logits, labels):
            log_flat, lab_flat = flatten_binary_scores(log, lab)
            loss.append(lovasz_hinge_flat(log_flat, lab_flat))
        return torch.mean(torch.stack(loss))
    else:
        log_flat, lab_flat = flatten_binary_scores(logits, labels)
        return lovasz_hinge_flat(log_flat, lab_flat)

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss flat
    logits: [P] flattened logits
    labels: [P] flattened labels
    """
    signs = 2.0 * labels.float() - 1.0
    errors = (1.0 - logits * signs)
    errors_sorted, perm = torch.sort(errors, descending=True)
    labels_sorted = labels[perm]
    grad = lovasz_grad(labels_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

# Loss: Focal Tversky for Classification
class ClassifierFocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss for imbalanced binary classification.
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=4./3., smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fn = ((1 - probs) * targets).sum()
        fp = (probs * (1 - targets)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return (1 - tversky) ** self.gamma

# Loss: Focal Tversky for Segmentation
class SegmentationFocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss: heavily penalizes FN over FP when α<β,
    and raises to a power γ to focus training on hard examples.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B,1,H,W) raw network outputs
        targets: (B,1,H,W) binary {0,1} masks
        """
        probs = torch.sigmoid(logits)
        # flatten spatial dims
        p = probs.view(probs.size(0), -1)
        t = targets.view(targets.size(0), -1)

        tp = (p * t).sum(dim=1)
        fn = ((1 - p) * t).sum(dim=1)
        fp = (p * (1 - t)).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )
        # focal term
        loss = (1 - tversky) ** self.gamma
        return loss.mean()

# Loss: Combined Focal-Tversky + Lovasz hinge
class CombinedSegmentationLoss(nn.Module):
    """
    Combines Focal-Tversky with a small Lovász hinge term.
    """
    def __init__(self,
                 ft_kwargs: dict = None,
                 lovasz_weight: float = 0.2):
        """
        ft_kwargs:    passed to FocalTverskyLoss(...)
        lovasz_weight: weight given to the lovasz term
        """
        super().__init__()
        ft_kwargs = ft_kwargs or {}
        self.ft_loss       = SegmentationFocalTverskyLoss(**ft_kwargs)
        self.lovasz_weight = lovasz_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B,1,H,W)
        targets: (B,1,H,W)
        """
        # make sure targets is float tensor
        t = targets.float()

        # 1) Focal-Tversky
        ft = self.ft_loss(logits, t)

        # 2) Lovász hinge expects shape (B,H,W) and logits
        #    so we squeeze out the channel dim:
        lov = lovasz_hinge(logits, t.squeeze(1))

        return ft + self.lovasz_weight * lov

def remove_small_regions_batch(preds: torch.Tensor, min_size: int = 500) -> torch.Tensor:
    """
    preds: (N,1,H,W) binary (0/1) torch tensor
    removes components smaller than `min_size` pixels
    """
    device = preds.device
    out = preds.clone()
    N,_,H,W = preds.shape
    for i in range(N):
        mask = (preds[i,0].cpu().numpy() > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_size:
                clean[labels == lab] = 1
        out[i,0] = torch.from_numpy(clean).to(device)
    return out