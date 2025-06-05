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

class OHEMLoss(nn.Module):
    def __init__(self, keep_ratio: float = 0.5, eps: float = 1e-6):
        super().__init__()
        assert 0 < keep_ratio <= 1.0
        self.keep_ratio = keep_ratio
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, tissue_mask: torch.Tensor) -> torch.Tensor:
        """
        logits:      (B,1,H,W)
        targets:     (B,1,H,W) in {0,1}
        tissue_mask: (B,1,H,W) in {0,1}
        """
        bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # (B,1,H,W)
        bce_map = bce_map * tissue_mask  # ignore background

        B, C, H, W = bce_map.shape
        flat_loss   = bce_map.view(B, -1)             # (B, H*W)
        flat_tgt    = targets.view(B, -1)             # (B, H*W)
        flat_tissue = tissue_mask.view(B, -1)         # (B, H*W)

        # Count how many pixels are positive inside tissue, per image
        pos_counts = (flat_tgt * flat_tissue).sum(dim=1).long()  # (B,)

        # Compute k = number of pixels to keep per image
        total_pixels = flat_loss.size(1)  # H*W
        k_all = max(1, int(total_pixels * self.keep_ratio))

        losses = []
        for i in range(B):
            loss_i   = flat_loss[i]   # shape (H*W,)
            tgt_i    = flat_tgt[i]
            tissue_i = flat_tissue[i]
            pos_i    = (tgt_i * tissue_i).nonzero().view(-1)  # indices of positive pixels

            # Always keep every positive pixel inside tissue
            keep_indices = pos_i.tolist()

            # Now pick the hardest negatives among the rest
            n_pos = pos_i.numel()
            n_remain = max(0, k_all - n_pos)

            if n_remain > 0:
                # Mask out positives so we only rank negatives
                neg_mask = ((tgt_i == 0) & (tissue_i == 1))
                neg_losses = loss_i[neg_mask]  # BCE of all negative‐tissue pixels

                # Clamp n_remain so we never ask for more negatives than exist
                num_neg = neg_losses.numel()
                n_remain_clamped = min(n_remain, num_neg)

                if n_remain_clamped > 0:
                    # top n_remain_clamped largest negative losses
                    topk_vals_neg, topk_idx_neg = neg_losses.topk(n_remain_clamped, largest=True, sorted=False)
                    # But topk_idx_neg are indices inside neg_losses; map back to original flatten index:
                    neg_flat_idx = neg_mask.nonzero().view(-1)  # these are the flat indices
                    chosen_negatives = neg_flat_idx[topk_idx_neg]  # original flat indices
                    keep_indices += chosen_negatives.tolist()

            # If there were zero positives and zero negatives inside tissue, keep one pixel
            if len(keep_indices) == 0:
                # select the overall pixel (inside tissue) with max loss
                tissue_indices = (tissue_i == 1).nonzero().view(-1)
                if tissue_indices.numel() > 0:
                    losses_in_tissue = loss_i[tissue_indices]
                    max_idx = losses_in_tissue.argmax().item()
                    keep_indices = [int(tissue_indices[max_idx])]
                else:
                    # no tissue at all—keep one pixel arbitrarily (the first)
                    keep_indices = [0]

            kept_losses = loss_i[keep_indices]
            losses.append(kept_losses.mean())

        return torch.stack(losses).mean()


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

# Loss: Combined Focal-Tversky + Lovasz hinge + OHEM
class CombinedSegmentationLoss(nn.Module):
    def __init__(self,
                 ft_kwargs: dict = None,
                 lovasz_weight: float = 0.2,
                 ohem_keep_ratio: float = 0.3):
        super().__init__()
        self.ohem_loss     = OHEMLoss(keep_ratio=ohem_keep_ratio)
        self.ft_loss       = SegmentationFocalTverskyLoss(**(ft_kwargs or {}))
        self.lovasz_weight = lovasz_weight

    def forward(self, logits, targets, tissue_mask):
        """
        logits:      (B,1,H,W)
        targets:     (B,1,H,W)
        tissue_mask: (B,1,H,W)
        """
        t = targets.float()
        # 1) Compute OHEM on all images (even if no positives—our new OHEM handles that)
        ohem_term = self.ohem_loss(logits, t, tissue_mask)

        # 2) For FT and Lovasz, only accumulate on images where there is at least one positive
        flat_t = t.view(t.size(0), -1)
        positive_batches = (flat_t.sum(dim=1) > 0)  # boolean mask of shape (B,)

        if positive_batches.any():
            ft_term = self.ft_loss(logits[positive_batches], targets[positive_batches])
            lov_term = lovasz_hinge(
                logits[positive_batches],
                targets[positive_batches].squeeze(1).long()
            )
            combined = ohem_term + ft_term + self.lovasz_weight * lov_term
        else:
            # all images in batch have zero positives → skip FT & Lovasz
            combined = ohem_term

        return combined


def remove_small_regions_batch(
    preds: torch.Tensor,
    tissue: torch.Tensor,
    min_size: int
) -> torch.Tensor:
    """
    preds:   (N,1,H,W)  binary torch tensor (0/1) of predicted mask
    tissue:  (N,1,H,W)  binary torch tensor mask of valid tissue
    min_size: int       minimum connected‐component size (in pixels) to keep
    
    Returns a tensor of shape (N,1,H,W) where any connected component
    that lies within 'tissue' and has area < min_size is removed (set to 0).
    """
    # Ensure both inputs are 4D
    if preds.dim() != 4 or tissue.dim() != 4:
        raise ValueError(f"remove_small_regions_batch expects 4D tensors, got preds.dim()={preds.dim()}, tissue.dim()={tissue.dim()}")

    device = preds.device
    dtype = preds.dtype
    out = preds.clone()

    N, C, H, W = preds.shape
    if C != 1:
        raise ValueError(f"remove_small_regions_batch expects preds.shape[1] == 1, got {C}")

    for i in range(N):
        # Convert this batch element to a CPU numpy array of 0/1
        p_np = (preds[i, 0].cpu().numpy() > 0).astype(np.uint8)
        t_np = (tissue[i, 0].cpu().numpy() > 0).astype(np.uint8)

        # Only process pixels inside tissue
        mask = p_np * t_np  # shape (H, W), 0/1

        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        clean = np.zeros_like(mask, dtype=np.uint8)

        # Keep only components whose area >= min_size
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area >= min_size:
                clean[labels == lab] = 1

        # Write back into out tensor, preserving dtype and device
        out[i, 0] = (
            torch.from_numpy(clean)
            .to(device)
            .to(dtype)
        )

    return out
