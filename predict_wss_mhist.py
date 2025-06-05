import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from torchvision import models, transforms
from torchvision.models import ResNet34_Weights
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
from torch.optim import SGD
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchcam.methods import SmoothGradCAMpp
from torch.amp import autocast, GradScaler
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_recall_curve, auc
from skimage.morphology import remove_small_objects
from PIL import Image
import copy
import torch.nn.functional as F

from datasets.mhist import get_mhist_loaders, MHISTDataset
from config.config_manager import ConfigManager
from metrics.wandb_logger import WandbLogger
from utils.visualization import visualize_segmentation_results
from utils.utils import ClassifierFocalTverskyLoss, CombinedSegmentationLoss, OHEMLoss, remove_small_regions_batch


# -----------------------------------
# Classifier Training (accepts loaders)
# -----------------------------------
def train_classifier(model: nn.Module,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     cfg: ConfigManager,
                     device: torch.device):
    """
    Train classification backbone given pre-built train_loader and val_loader.
    Returns best_model_state_dict.
    """
    bce_loss = nn.BCEWithLogitsLoss()
    ft_loss = ClassifierFocalTverskyLoss(
        alpha=cfg.get("classification.tversky_alpha"),
        beta=cfg.get("classification.tversky_beta"),
        gamma=cfg.get("classification.tversky_gamma"),
    ).to(device)

    def loss_fn(logits, labels):
        labels_f = labels.float().unsqueeze(1).to(device)
        return bce_loss(logits, labels_f) + ft_loss(logits, labels_f)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.get("classification.lr"),
        weight_decay=cfg.get("classification.weight_decay"),
        momentum=0.9,
    )

    total_epochs = sum(phase["epochs"] for phase in cfg.get("classification.unfreeze_schedule"))
    total_steps = total_epochs * len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.get("classification.lr"),
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="linear",
    )

    best_val_acc = 0.0
    best_state = None
    fixed_thresh = cfg.get("classification.threshold")
    global_step = 0
    epoch_counter = 0

    for phase_idx, phase in enumerate(cfg.get("classification.unfreeze_schedule")):
        layers_to_unfreeze = phase["layers"]
        epochs = phase["epochs"]

        # Freeze all layers
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze head + specified layers
        for name, param in model.named_parameters():
            if name.startswith("fc.") or any(name.startswith(f"layer{li}") for li in layers_to_unfreeze):
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Classifier] Phase {phase_idx+1}: unfreezing {layers_to_unfreeze}, {n_trainable} trainable params")

        for _ in range(epochs):
            epoch_counter += 1
            model.train()
            running_loss = 0.0
            running_acc = 0.0

            for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch_counter}/{total_epochs}", ncols=80):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                logits = model(imgs)
                loss = loss_fn(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds = (probs > fixed_thresh).float()
                    lbls = labels.float().unsqueeze(1)
                    running_acc += (preds == lbls).float().mean().item()
                running_loss += loss.item()

            n_batches = len(train_loader)
            print(f"[Clf] Loss={running_loss/n_batches:.4f} Acc={running_acc/n_batches:.4f}")

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    probs = torch.sigmoid(model(imgs))
                    preds = (probs > fixed_thresh).float()
                    lbls = labels.float().unsqueeze(1)
                    val_correct += (preds == lbls).float().sum().item()
                    val_total += preds.numel()
            val_acc = val_correct / val_total
            print(f"[Clf] Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, cfg.get("classification.save_path"))
                print(f"[Classifier] Saved best Acc: {best_val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Classifier] Restored best model @ Acc={best_val_acc:.4f}")
    return model


# -----------------------------------
# Pseudo-mask Generation
# -----------------------------------
def extract_pseudo_masks(model: nn.Module,
                         dataset: MHISTDataset,
                         cfg: ConfigManager,
                         device: torch.device):
    """
    Generate CAM-based pseudo-masks for SSA images, apply morphology + CRF, save to data.mask_dir.
    """
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

    cam_size = tuple(cfg.get("postprocessing.cam_resize", [224, 224]))
    mean = cfg.get("postprocessing.cam_normalize_mean", [0.485, 0.456, 0.406])
    std = cfg.get("postprocessing.cam_normalize_std", [0.229, 0.224, 0.225])
    cam_tf = transforms.Compose([
        transforms.Resize(cam_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    out_dir = cfg.get("data.mask_dir")
    os.makedirs(out_dir, exist_ok=True)

    thresh_q     = cfg.get("postprocessing.threshold_quantile", 0.25)
    open_k       = cfg.get("postprocessing.open_kernel", 3)
    close_k      = cfg.get("postprocessing.close_kernel", 5)
    min_obj      = cfg.get("postprocessing.min_object_size", 64)
    crf_iters    = cfg.get("postprocessing.crf_iters", 3)
    gauss_sxy    = cfg.get("postprocessing.gaussian_sxy", 3)
    gauss_compat = cfg.get("postprocessing.gaussian_compat", 3)
    bilat_sxy    = cfg.get("postprocessing.bilateral_sxy", 80)
    bilat_srgb   = cfg.get("postprocessing.bilateral_srgb", 13)
    bilat_compat = cfg.get("postprocessing.bilateral_compat", 10)

    print("[CAM] Generating pseudo-masks...")
    for idx in tqdm(range(len(dataset)), desc="CAM Extract", ncols=80):
        fname = dataset.data.iloc[idx]["Image Name"]
        pil = Image.open(os.path.join(cfg.get("data.img_dir"), fname)).convert("RGB")
        W, H = pil.size
        orig = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        _, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = (bg == 0).astype(np.uint8)

        _, label = dataset[idx]  # 0=HP, 1=SSA

        inp = cam_tf(pil).unsqueeze(0).to(device)
        model.zero_grad()
        logits = model(inp)

        if label == 1:
            cam = cam_extractor(0, logits)[0].squeeze().cpu().numpy()
            cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
        else:
            cam = np.zeros((H, W), dtype=np.float32)

        cam *= tissue_mask
        q = np.quantile(cam[tissue_mask == 1], thresh_q) if label == 1 else 1.0
        mask = ((cam > q) & (tissue_mask == 1)).astype(np.uint8)

        ko = np.ones((open_k, open_k), np.uint8)
        kc = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        mask = remove_small_objects(mask.astype(bool), min_size=min_obj).astype(np.uint8)

        prob = np.stack([1 - mask, mask], axis=0)
        U = unary_from_softmax(prob.reshape(2, -1))
        if U.shape != (2, H * W):
            raise ValueError(f"Unary shape {U.shape}, expected (2, {H*W})")

        dcrf_model = dcrf.DenseCRF2D(W, H, 2)
        dcrf_model.setUnaryEnergy(U)
        dcrf_model.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
        dcrf_model.addPairwiseBilateral(sxy=bilat_sxy, srgb=bilat_srgb, rgbim=orig, compat=bilat_compat)
        Q = np.array(dcrf_model.inference(crf_iters))
        refined = Q.argmax(axis=0).reshape(H, W).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + "_mask.png"
        cv2.imwrite(os.path.join(out_dir, out_name), refined * 255)

    print("[CAM] Pseudo-masks done.")


# -----------------------------------
# Segmentation Evaluation
# -----------------------------------
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve, auc
from utils.utils import remove_small_regions_batch

def evaluate_segmentation(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    cfg: ConfigManager,
    device: torch.device
):
    """
    Evaluate segmentation: per-image IoU and global PR AUC (both restricted to 'tissue' pixels).
    - Removes connected components smaller than cfg.get('postprocessing.min_region', 500) before IoU.
    """

    model.eval()
    all_probs_list = []    # list of 1D numpy arrays (one per image) of probability scores inside tissue
    all_labels_list = []   # same shape, but ground‐truth
    per_image_ious = []

    threshold  = cfg.get("training.threshold", 0.5)
    min_region = cfg.get("postprocessing.min_region", 500)

    with torch.no_grad():
        for imgs, gts, tissues in val_loader:
            # imgs:    (B, 3, H, W)
            # gts:     (B, 1, H, W)  {0,1}
            # tissues: (B, 1, H, W)  {0,1}
            imgs     = imgs.to(device, non_blocking=True)
            gts      = gts.to(device, non_blocking=True)
            tissues  = tissues.to(device, non_blocking=True)

            # Forward
            logits = model(imgs)                # (B,1,H,W)
            probs  = torch.sigmoid(logits)      # (B,1,H,W)

            B, C, H, W = probs.shape
            # Flatten each tensor to shape (B, H*W)
            flat_probs   = probs.view(B, -1).cpu().numpy()             # float32
            flat_labels  = gts.view(B, -1).cpu().numpy().astype(np.uint8)
            flat_tissue  = tissues.view(B, -1).cpu().numpy().astype(np.uint8)

            for i in range(B):
                mask_i = flat_tissue[i].astype(bool)
                if mask_i.sum() == 0:
                    # No tissue at all in this slide → skip PR & IoU entirely
                    continue

                # 1) Append probabilities & labels (only where tissue==1) for global PR curve
                all_probs_list.append(flat_probs[i][mask_i])
                all_labels_list.append(flat_labels[i][mask_i])

                # 2) Per‐image IoU (only on SSA slides: where GT sum>0 inside tissue)
                l_i = flat_labels[i][mask_i]
                if l_i.sum() == 0:
                    # Pure HP (no lesion) → skip IoU
                    continue

                p_i = flat_probs[i][mask_i]
                pred_bin = (p_i > threshold).astype(np.uint8)

                # Remove small spurious islands:
                # Build a (1,1,H_tissue) binary tensor & dummy tissue mask of same shape.
                # First reconstruct a full‐size binary mask (H×W) with zeros outside tissue positions:
                full_pred = np.zeros((H * W,), dtype=np.uint8)
                full_pred[mask_i] = pred_bin
                full_pred = full_pred.reshape(H, W)

                # Use remove_small_regions_batch on a 4D tensor of shape (1,1,H,W)
                tmp_pred    = torch.from_numpy(full_pred).unsqueeze(0).unsqueeze(0).float().to(device)
                tmp_tissue  = tissues[i, 0].unsqueeze(0).unsqueeze(0)  # already 0/1 on the device

                clean_pred_tensor = remove_small_regions_batch(tmp_pred, tmp_tissue, min_region)
                clean_pred_np = clean_pred_tensor.view(-1).cpu().numpy().astype(np.uint8)[mask_i]

                # Now compute IoU on tissue pixels
                inter = (clean_pred_np * l_i).sum()
                union = ((clean_pred_np + l_i) >= 1).sum()
                per_image_ious.append(inter / (union + 1e-6))

    # --- Global PR AUC over concatenated tissue‐only pixels ---
    if all_probs_list:
        flat_probs_all  = np.concatenate(all_probs_list)
        flat_labels_all = np.concatenate(all_labels_list)
        prec, rec, ths = precision_recall_curve(flat_labels_all, flat_probs_all)
        pr_auc = auc(rec, prec)
    else:
        pr_auc = 0.0

    # --- Mean IoU over all SSA slides in validation ---
    if per_image_ious:
        mean_iou = float(np.mean(per_image_ious))
        std_iou  = float(np.std(per_image_ious))
        n_iou    = len(per_image_ious)
    else:
        mean_iou = 0.0
        std_iou  = 0.0
        n_iou    = 0

    print(f"Per-image IoU @th={threshold:.2f}: mean={mean_iou:.4f}, std={std_iou:.4f}, n={n_iou}")
    print(f"Global PR AUC: {pr_auc:.4f}")

    return {
        "per_image_ious": np.array(per_image_ious),
        "precision":     prec if all_probs_list else np.array([]),
        "recall":        rec if all_probs_list else np.array([]),
        "thresholds":    ths if all_probs_list else np.array([]),
        "pr_auc":        pr_auc,
    }


# -----------------------------------
# Segmentation Training (accepts loaders)
# -----------------------------------
def train_segmentation(
    model: nn.Module,
    ssa_loader: torch.utils.data.DataLoader,
    hp_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: ConfigManager,
    device: torch.device,
    logger=None
):
    """
    Train segmentation with:
      - Dice + Lovász (masked by tissue)
      - Slide-level classification head (low weight)
      - Progressive introduction of HP (negatives) up to p_hp=0.3
      - Validation masked by tissue, threshold fixed at 0.5
      - Logging of PR AUC and mean IoU on validation
      - Save model whenever mean IoU at 0.5 improves
    """
    # 1) Loss functions
    dice_loss_f = smp.losses.DiceLoss(mode="binary").to(device)
    lovasz_loss_f = smp.losses.LovaszLoss(mode="binary").to(device)

    # 2) Optimizer & Scheduler
    optimizer = SGD(
        model.parameters(),
        lr=cfg.get("training.learning_rate", 5e-4),
        momentum=0.9,
        weight_decay=cfg.get("training.weight_decay", 1e-4),
    )
    steps_per_epoch = len(ssa_loader) + len(hp_loader)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #    optimizer,
    #    T_max=cfg.get("training.epochs"),
    #    eta_min=cfg.get("training.min_lr", 1e-5),
    #    last_epoch=-1,
    #)
    scaler = GradScaler()

    # 3) Hyperparameters
    warmup_ep = cfg.get("segmentation.train.warmup_epochs",
                        max(1, cfg.get("training.epochs") // 3))
    val_thresh = 0.5
    min_region = cfg.get("postprocessing.min_region", 500)
    cls_weight = cfg.get("segmentation.cls_loss_weight", 0.2)

    # Track the best mean IoU across all epochs
    best_iou_so_far = 0.0

    for epoch in range(1, cfg.get("training.epochs") + 1):
        model.train()
        epoch_loss = 0.0

        # Progressive p_hp: 0 during warmup, then linearly up to 0.3 over next warmup_ep epochs
        if epoch <= warmup_ep:
            p_hp = 0.0
        else:
            frac = (epoch - warmup_ep) / warmup_ep
            p_hp = min(0.3, frac * 0.3)

        ssa_iter = iter(ssa_loader)
        hp_iter = iter(hp_loader)

        # --- Training loop over one “epoch” (all SSA + all HP) ---
        for _ in tqdm(
            range(steps_per_epoch),
            desc=f"[Seg] Ep{epoch}/{cfg.get('training.epochs')}",
            ncols=80
        ):
            # Choose HP with probability p_hp, otherwise SSA
            if torch.rand(1).item() < p_hp:
                try:
                    imgs, gts, tissues = next(hp_iter)
                except StopIteration:
                    hp_iter = iter(hp_loader)
                    imgs, gts, tissues = next(hp_iter)
            else:
                try:
                    imgs, gts, tissues = next(ssa_iter)
                except StopIteration:
                    ssa_iter = iter(ssa_loader)
                    imgs, gts, tissues = next(ssa_iter)

            imgs    = imgs.to(device, non_blocking=True)     # (B,3,H,W)
            gts     = gts.to(device, non_blocking=True)      # (B,1,H,W) 0/1
            tissues = tissues.to(device, non_blocking=True)  # (B,1,H,W) 0/1

            optimizer.zero_grad()
            with autocast(device_type=device.type):
                seg_logits = model(imgs)            # (B,1,H,W)
                probs      = torch.sigmoid(seg_logits)

                B, C, H, W = seg_logits.shape
                # Flatten
                flat_logits = seg_logits.view(B, -1)    # (B, H*W)
                flat_probs  = probs.view(B, -1)         # (B, H*W)
                flat_gts    = gts.view(B, -1)           # (B, H*W)
                flat_tissue = tissues.view(B, -1)       # (B, H*W)

                # 3.1) Masked Dice (only inside tissue)
                masked_probs = (flat_probs * flat_tissue).view(B, 1, H, W)
                masked_gts   = (flat_gts   * flat_tissue).view(B, 1, H, W)
                loss_dice    = dice_loss_f(masked_probs, masked_gts)

                # 3.2) Masked Lovász (zero out logits outside tissue)
                logits_masked = (flat_logits * flat_tissue).view(B, 1, H, W)
                loss_lovasz   = lovasz_loss_f(logits_masked, masked_gts)

                seg_loss = loss_dice + 0.5 * loss_lovasz

                # 3.3) Slide-level classification head (max over tissue)
                masked_logits = flat_logits * flat_tissue   # (B, H*W), zeros outside tissue
                cls_log = masked_logits.max(dim=1).values  # (B,)
                slide_lb = (masked_gts.view(B, -1).sum(dim=1) > 0).float().to(device)
                cls_loss = F.binary_cross_entropy_with_logits(cls_log, slide_lb)

                loss = seg_loss + cls_weight * cls_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_train = epoch_loss / steps_per_epoch

        #scheduler.step()
        current_lr = 0.001
        #current_lr = optimizer.param_groups[0]["lr"]

        print(f"[Seg] Epoch {epoch} Train Loss: {avg_train:.4f} LR: {current_lr:.6f} p_hp={p_hp:.2f}")

        if logger:
            logger.log_scalar("Train/Loss", avg_train, step=epoch)
            logger.log_scalar("Train/p_hp", p_hp, step=epoch)

        # --- Validation at fixed threshold 0.5 ---
        model.eval()
        all_probs_list  = []
        all_labels_list = []
        per_image_ious  = []

        with torch.no_grad():
            for imgs, gts_b, tissues_b in val_loader:
                imgs      = imgs.to(device, non_blocking=True)       # (B,3,H,W)
                gts_b     = gts_b.to(device, non_blocking=True)      # (B,1,H,W)
                tissues_b = tissues_b.to(device, non_blocking=True)  # (B,1,H,W)

                with autocast(device_type=device.type):
                    logits = model(imgs)               # (B,1,H,W)
                    probs  = torch.sigmoid(logits)     # (B,1,H,W)

                B, C, H, W = probs.shape
                flat_probs_b  = probs.view(B, -1).cpu().numpy()             # (B, H*W)
                flat_labels_b = gts_b.view(B, -1).cpu().numpy().astype(np.uint8)
                flat_tissue_b = tissues_b.view(B, -1).cpu().numpy().astype(np.uint8)

                for i in range(B):
                    mask_i = flat_tissue_b[i].astype(bool)
                    if mask_i.sum() == 0:
                        # No tissue → skip this slide entirely
                        continue

                    # 1) Append for global PR‐AUC
                    all_probs_list.append(flat_probs_b[i][mask_i])
                    all_labels_list.append(flat_labels_b[i][mask_i])

                    # 2) If SSA (some positive inside tissue), compute IoU at threshold=0.5
                    l_i = flat_labels_b[i][mask_i]
                    if l_i.sum() == 0:
                        # Pure HP → skip IoU
                        continue

                    p_i = flat_probs_b[i][mask_i]  # probabilities inside tissue
                    pred_bin_tissue = (p_i > val_thresh).astype(np.uint8)

                    # Reconstruct full (H×W) mask
                    full_pred_flat = np.zeros((H * W,), dtype=np.uint8)
                    full_pred_flat[mask_i] = pred_bin_tissue
                    full_pred = full_pred_flat.reshape(H, W)       # (H, W)

                    # Convert to 4D tensors (1,1,H,W) for small-region removal
                    tmp_pred   = torch.from_numpy(full_pred).unsqueeze(0).unsqueeze(0).float().to(device)
                    tmp_tissue = tissues_b[i, 0].unsqueeze(0).unsqueeze(0).float().to(device)

                    clean_pred_tensor = remove_small_regions_batch(tmp_pred, tmp_tissue, min_region)
                    clean_flat = clean_pred_tensor.view(-1).cpu().numpy().astype(np.uint8)[mask_i]

                    # Compute IoU inside tissue
                    inter = (clean_flat * l_i).sum()
                    union = ((clean_flat + l_i) >= 1).sum()
                    per_image_ious.append(inter / (union + 1e-6))

        # ——— Compute mean IoU at 0.5 ———
        if per_image_ious:
            mean_iou = float(np.mean(per_image_ious))
            std_iou  = float(np.std(per_image_ious))
            n_iou    = len(per_image_ious)
        else:
            mean_iou = 0.0
            std_iou  = 0.0
            n_iou    = 0

        print(f"[Seg] Validation @th={val_thresh:.2f}: mean IoU={mean_iou:.4f}, std={std_iou:.4f}, n={n_iou}")

        if logger:
            logger.log_scalar("Eval/IoU", mean_iou, step=epoch)

        # ——— Model saving ———
        if mean_iou > best_iou_so_far:
            best_iou_so_far = mean_iou
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_threshold": val_thresh,
                    "epoch": epoch,
                    "best_iou": best_iou_so_far
                },
                cfg.get("model.saved_path")
            )
            print(f"[Seg] Saved new best model @ Epoch {epoch}: IoU={best_iou_so_far:.4f} (th={val_thresh:.2f})")

    return model

# -----------------------------------
# Main Pipeline
# -----------------------------------
def main():
    # --- Reproducibility setup ---
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensures deterministic behavior on GPU (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = ConfigManager("config/default_config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = WandbLogger(
        project=cfg.get("logging.project"),
        name=cfg.get("logging.run_name"),
        config=cfg.config,
    )

    # === Segmentation loaders ===
    seg_cfg = {
        "task": "segmentation",
        "data.csv_path": cfg.get("data.csv_path"),
        "data.img_dir": cfg.get("data.img_dir"),
        "data.mask_dir": cfg.get("data.mask_dir"),
        "data.batch_size": cfg.get("data.batch_size"),
        "data.num_workers": cfg.get("data.num_workers", 4),
        "augmentation": {"segmentation": cfg.get("augmentation.segmentation", {})}
    }
    ssa_loader, hp_loader, val_loader = get_mhist_loaders(seg_cfg)

    # Build segmentation model
    seg_model = smp.DeepLabV3Plus(
        encoder_name=cfg.get("model.backbone"),
        encoder_weights="imagenet" if cfg.get("model.pretrained_backbone") else None,
        in_channels=3,
        classes=1,
        activation=None,
        decoder_atrous_rates=cfg.get("model.aspp_dilate", [6, 12, 18]),
    ).to(device)

    # Train segmentation
    seg_model = train_segmentation(seg_model, ssa_loader, hp_loader, val_loader, cfg, device, logger)

    # === Final evaluation & visualization ===
    print("[Main] Final eval & visualization...")
    evaluate_segmentation(seg_model, val_loader, cfg, device)

    # === Test-time loader ===
    test_cfg = {
        "task": "segmentation",
        "data.csv_path": cfg.get("data.csv_path"),
        "data.img_dir": cfg.get("data.img_dir"),
        "data.mask_dir": cfg.get("data.mask_dir"),
        "data.batch_size": cfg.get("classification.batch_size"),
        "data.num_workers": cfg.get("data.num_workers", 4),
        "augmentation": {"segmentation": cfg.get("augmentation.segmentation", {})}
    }
    _, _, test_loader = get_mhist_loaders(test_cfg)

    with torch.no_grad():
        for imgs, msks, _ in test_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)
            outs = seg_model(imgs)
            probs = torch.sigmoid(outs)
            preds = (probs > cfg.get("training.threshold")).float()
            visualize_segmentation_results(
                images=imgs,
                masks=msks,
                predictions=preds,
                probabilities=probs,
                step=cfg.get("training.epochs"),
                logger=logger,
            )
            break


if __name__ == "__main__":
    main()
