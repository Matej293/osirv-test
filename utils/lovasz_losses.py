import torch
import torch.nn.functional as F

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
