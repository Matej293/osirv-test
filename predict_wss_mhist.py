import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.optim import SGD
from torchvision import models
from torchvision.models import ResNet34_Weights
from torchcam.methods import SmoothGradCAMpp
from torch.amp import autocast, GradScaler
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import segmentation_models_pytorch as smp
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from metrics.wandb_logger import WandbLogger
from utils.visualization import visualize_segmentation_results
import copy
from torch.optim.lr_scheduler import CyclicLR
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import remove_small_objects
from PIL import Image
from datasets.mhist import MHISTDataset
from sklearn.metrics import precision_recall_curve, auc
from utils.utils import *

# -----------------------------------
# Classifier Pretraining
# -----------------------------------
def pretrain_classifier(cfg, device):
    """Pre-train the MHIST classification backbone with gradual unfreezing and a single OneCycleLR."""
    # 1) Build train+val loaders
    train_loader = get_mhist_dataloader(
        csv_file=cfg.get('data.csv_path'),
        img_dir=cfg.get('data.img_dir'),
        mask_dir=None,
        batch_size=cfg.get('classification.batch_size'),
        partition='train',
        augmentation_config=cfg.get('augmentation.classification'),
        task='classification'
    )
    val_loader = get_mhist_dataloader(
        csv_file=cfg.get('data.csv_path'),
        img_dir=cfg.get('data.img_dir'),
        mask_dir=None,
        batch_size=cfg.get('classification.batch_size'),
        partition='test',
        augmentation_config=cfg.get('augmentation.classification'),
        task='classification'
    )

    # 1a) Weighted sampler for train set
    train_ds = train_loader.dataset
    labels = train_ds.data['label'].astype(int).values
    counts = np.bincount(labels)
    sample_weights = 1.0 / counts[labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.get('classification.batch_size'),
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # 2) Build backbone + MLP head
    backbone_name = cfg.get('classification.backbone')
    backbone_cls = getattr(models, backbone_name)
    backbone = backbone_cls(weights=ResNet34_Weights.DEFAULT)
    in_features = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(128, 1),
    )
    model = backbone.to(device)

    # 3) Loss functions
    bce_loss = nn.BCEWithLogitsLoss()
    ft_loss = ClassifierFocalTverskyLoss(
        alpha=cfg.get('classification.tversky_alpha'),
        beta=cfg.get('classification.tversky_beta'),
        gamma=cfg.get('classification.tversky_gamma')
    ).to(device)
    def loss_fn(logits, labels):
        labels_f = labels.float().unsqueeze(1).to(device)
        return bce_loss(logits, labels_f) + ft_loss(logits, labels_f)

    # 4) Optimizer; initial lr will be set by OneCycleLR (max_lr = cfg.lr)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.get('classification.lr'),
        weight_decay=cfg.get('classification.weight_decay'),
        momentum=0.9,
    )

    # 5) Compute total steps for OneCycleLR
    total_epochs = sum(phase['epochs'] for phase in cfg.get('classification.unfreeze_schedule'))
    total_steps = total_epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.get('classification.lr'),
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    best_val_acc = 0.0
    best_state = None
    fixed_thresh = cfg.get('classification.threshold')
    global_step = 0
    epoch_counter = 0

    # 6) Gradual unfreezing loop
    for phase_idx, phase in enumerate(cfg.get('classification.unfreeze_schedule')):
        layers_to_unfreeze = phase['layers']
        epochs = phase['epochs']

        # Freeze all, then unfreeze head + specified layers
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith('fc.') or any(name.startswith(f'layer{layer_idx}') for layer_idx in layers_to_unfreeze):
                param.requires_grad = True

        # Count trainable parameters
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Classifier] Phase {phase_idx+1}: unfreezing layers {layers_to_unfreeze}, {n_trainable} params trainable")

        # Train for this phase
        for _ in range(epochs):
            epoch_counter += 1
            model.train()
            running_loss = 0.0
            running_acc = 0.0

            for imgs, labels in tqdm(train_loader, desc=f"Phase{phase_idx+1} Epoch{epoch_counter}/{total_steps//len(train_loader)}"):
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
            print(f"[Clf] Phase{phase_idx+1} "
                  f"Loss={running_loss/n_batches:.4f} "
                  f"Acc={running_acc/n_batches:.4f}")

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
            print(f"[Clf]    Val Acc={val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, cfg.get('classification.save_path'))
                print(f"[Classifier] Saved best Acc: {best_val_acc:.4f}")

    # 7) Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Classifier] Restored best model @ Acc={best_val_acc:.4f}")

    return model, train_ds


# -----------------------------------
# Pseudo-mask Generation
# -----------------------------------
def extract_pseudo_masks(model, dataset, config, device):
    """
    Batched CAM → expanded morphology → small-object pruning → CRF refinement.
    Saves refined masks into config.get('data.mask_dir').
    """
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")
    # build the CAM‐preprocessing transform from config
    cam_size = tuple(config.get("postprocessing.cam_resize", [224, 224]))
    mean = config.get("postprocessing.cam_normalize_mean", [0.485, 0.456, 0.406])
    std  = config.get("postprocessing.cam_normalize_std",  [0.229, 0.224, 0.225])
    cam_tf = transforms.Compose([
        transforms.Resize(cam_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    out_dir = config.get("data.mask_dir")
    os.makedirs(out_dir, exist_ok=True)
    thresh_q       = config.get("postprocessing.threshold_quantile", 0.25)
    open_k         = config.get("postprocessing.open_kernel", 3)
    close_k        = config.get("postprocessing.close_kernel", 5)
    min_obj        = config.get("postprocessing.min_object_size", 64)
    crf_iters      = config.get("postprocessing.crf_iters", 3)
    gauss_sxy      = config.get("postprocessing.gaussian_sxy", 3)
    gauss_compat   = config.get("postprocessing.gaussian_compat", 3)
    bilat_sxy      = config.get("postprocessing.bilateral_sxy", 80)
    bilat_srgb     = config.get("postprocessing.bilateral_srgb", 13)
    bilat_compat   = config.get("postprocessing.bilateral_compat", 10)
    print("[CAM] Generating pseudo-masks with background masking...")
    for idx in tqdm(range(len(dataset)), desc="CAM Extract"):
        fname = dataset.data.iloc[idx]["Image Name"]
        pil   = Image.open(os.path.join(config.get("data.img_dir"), fname)).convert("RGB")
        W, H  = pil.size
        orig = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        # 1) convert to gray
        gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        # 2) compute Otsu’s threshold
        _, bg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 3) tissue mask = “non‐background”
        tissue_mask = (bg == 0).astype(np.uint8)
        # fetch label to skip HP CAM
        _, label = dataset[idx]  # label: 0=HP, 1=SSA
        # forward through classifier
        inp    = cam_tf(pil).unsqueeze(0).to(device)
        model.zero_grad()
        logits = model(inp)
        if label == 1:
            cam = cam_extractor(0, logits)[0].squeeze().cpu().numpy()
            cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
        else:
            cam = np.zeros((H, W), dtype=np.float32)
        cam *= tissue_mask
        q    = np.quantile(cam[tissue_mask==1], thresh_q) if label == 1 else 1.0
        mask = ((cam > q) & (tissue_mask == 1)).astype(np.uint8)
        ko   = np.ones((open_k, open_k), np.uint8)
        kc   = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        mask = remove_small_objects(mask.astype(bool), min_size=min_obj).astype(np.uint8)
        # prepare unary for CRF
        prob = np.stack([1 - mask, mask], axis=0)
        U    = unary_from_softmax(prob.reshape(2, -1))
        if U.shape != (2, H*W):
            raise ValueError(f"Unary shape {U.shape}, expected (2, {H*W})")
        # setup DenseCRF
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
        d = dcrf.DenseCRF2D(W, H, 2)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=gauss_sxy, compat=gauss_compat)
        d.addPairwiseBilateral(
            sxy=bilat_sxy, srgb=bilat_srgb, rgbim=orig, compat=bilat_compat
        )
        Q = np.array(d.inference(crf_iters))
        refined = Q.argmax(axis=0).reshape(H, W).astype(np.uint8)
        out_name = os.path.splitext(fname)[0] + "_mask.png"
        cv2.imwrite(os.path.join(out_dir, out_name), refined * 255)
    print("[CAM] Pseudo-masks done.")

# -----------------------------------
# Segmentation Training
# -----------------------------------
from sklearn.metrics import precision_recall_curve, auc
import numpy as np
import torch

def evaluate_segmentation(model, val_loader, cfg, device):
    """
    Runs the model over val_loader (which must include both SSA and HP slides) and:
      1. Computes per-image IoU over the full mask (so false-positives on HP hurt you).
      2. Builds a global precision–recall curve over all pixels.
    """
    model.eval()
    all_probs      = []
    all_labels     = []
    per_image_ious = []
    threshold = cfg.get('training.threshold', 0.5)
    with torch.no_grad():
        for imgs, gts, _ in val_loader:
            imgs = imgs.to(device)
            gts  = gts.to(device)           # (B,1,H,W)
            logits = model(imgs)            # (B,1,H,W)
            probs  = torch.sigmoid(logits)  # (B,1,H,W)
            B = probs.size(0)
            # for each slide in the batch
            for i in range(B):
                p = probs[i, 0].cpu().numpy().ravel()           # float32 array
                l = gts[i, 0].cpu().numpy().ravel().astype(np.uint8)  # uint8 array
                all_probs.append(p)
                all_labels.append(l)
                # per-image IoU at default threshold
                pred_bin = (p > threshold).astype(np.uint8)
                inter    = (pred_bin * l).sum()
                union    = ((pred_bin + l) >= 1).sum()
                per_image_ious.append(inter / (union + 1e-6))
    # concatenate for global PR
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    # compute PR curve & AUC
    prec, rec, ths = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec, prec)
    # report per-image IoU distribution
    ious = np.array(per_image_ious)
    print(f"Per-image IoU @th={threshold:.2f}: "
          f"mean = {ious.mean():.4f}, std = {ious.std():.4f}, n = {len(ious)}")
    print(f"Global PR AUC: {pr_auc:.4f}")
    return {
        "per_image_ious": ious,
        "precision":     prec,
        "recall":        rec,
        "thresholds":    ths,
        "pr_auc":        pr_auc,
    }

def make_seg_loaders(cfg):
    resize = tuple(cfg.get('augmentation.segmentation.train.resize', (224, 224)))
    dummy_tf = transforms.Compose([transforms.Resize(resize)])
    # full train dataset, contains both SSA & HP
    full_train_ds = MHISTDataset(
        csv_file = cfg.get('data.csv_path'),
        img_dir  = cfg.get('data.img_dir'),
        mask_dir = cfg.get('data.mask_dir'),
        transform=dummy_tf,
        partition='train',
        task='segmentation'
    )
    # build SSA‐only subset
    is_ssa = full_train_ds.data["Majority Vote Label"] == "SSA"
    ssa_indices = np.where(is_ssa.values)[0].tolist()
    if not ssa_indices:
        raise RuntimeError("No SSA samples in train partition!")
    ssa_train_ds = Subset(full_train_ds, ssa_indices)
    # test/val loader
    test_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=cfg.get('augmentation.segmentation.test.mean'),
            std=cfg.get('augmentation.segmentation.test.std')
        )
    ])
    val_ds = MHISTDataset(
        csv_file = cfg.get('data.csv_path'),
        img_dir  = cfg.get('data.img_dir'),
        mask_dir = cfg.get('data.mask_dir'),
        transform=test_tf,
        partition='test',
        task='segmentation'
    )
    # alb augment for training
    alb = cfg.get('augmentation.segmentation.train')
    train_aug = A.Compose([
        TorchstainNormalize(p=alb.get('stain_normalize_p', 0.7)),
        A.Resize(*resize),
        A.HorizontalFlip(p=alb.get('h_flip_p', 0.5)),
        A.VerticalFlip(p=alb.get('v_flip_p', 0.5)),
        A.RandomRotate90(p=alb.get('rot90_p', 0.5)),
        A.ElasticTransform(
            p=alb.get('elastic_p', 0.3),
            alpha=alb.get('elastic_alpha', 1.0),
            sigma=alb.get('elastic_sigma', 50)
        ),
        A.GridDistortion(p=alb.get('grid_p', 0.3)),
        A.HueSaturationValue(
            p=alb.get('hsv_p', 0.3),
            hue_shift_limit=alb.get('hue', 10),
            sat_shift_limit=alb.get('sat', 30),
            val_shift_limit=alb.get('val', 10)
        ),
        A.Normalize(mean=alb.get('mean'), std=alb.get('std')),
        ToTensorV2(),
    ], additional_targets={'tissue': 'mask'})
    # wrapper to re-open original image for albumentations
    class TrainWrapper(torch.utils.data.Dataset):
        def __init__(self, base_ds, aug, img_dir):
            self.base   = base_ds
            self.aug    = aug
            self.img_dir= img_dir
        def __len__(self):
            return len(self.base)
        def __getitem__(self, i):
            # map Subset -> raw_ds, real_idx
            if isinstance(self.base, Subset):
                real_idx = self.base.indices[i]
                raw_ds   = self.base.dataset
            else:
                real_idx = i
                raw_ds   = self.base
            row   = raw_ds.data.iloc[real_idx]
            pil   = Image.open(os.path.join(self.img_dir, row["Image Name"])).convert("RGB")
            img_np= np.array(pil)
            _, gt_mask, tissue_mask = raw_ds[real_idx]
            mask_np     = (gt_mask.squeeze(0).numpy() * 255).astype(np.uint8)
            tissue_np   = (tissue_mask.squeeze(0).numpy() * 255).astype(np.uint8)
            d = self.aug(image=img_np, mask=mask_np, tissue=tissue_np)
            return (
                d['image'],
                d['mask'][None].float() / 255.0,
                d['tissue'][None].float() / 255.0
            )
    batch_size = cfg.get('data.batch_size')
    ssa_loader = DataLoader(
        TrainWrapper(ssa_train_ds, train_aug, cfg.get('data.img_dir')),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    full_loader = DataLoader(
        TrainWrapper(full_train_ds, train_aug, cfg.get('data.img_dir')),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"[SegLoader] SSA-only: {len(ssa_train_ds)}, full-train: {len(full_train_ds)}, val: {len(val_ds)}")
    return ssa_loader, full_loader, val_loader

def train_segmentation(cfg, device, logger=None, step=None):
    """
    Train DeepLabV3+ with gradual hard‐negative mining, CLMR, and
    a tiny classification head on top of the segmentation logits.
    """
    # 1) loaders
    ssa_loader, full_loader, val_loader = make_seg_loaders(cfg)
    # 2) Segmentation model as before
    model = smp.DeepLabV3Plus(
        encoder_name    = cfg.get('model.backbone'),
        encoder_weights = 'imagenet' if cfg.get('model.pretrained_backbone') else None,
        in_channels     = 3,
        classes         = 1,
        activation      = None,
        decoder_atrous_rates = cfg.get('model.aspp_dilate', [6, 12, 18]),
    ).to(device)
    # 3) Losses & Optimizer
    criterion = CombinedSegmentationLoss(
        ft_kwargs       = dict(alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6),
        lovasz_weight   = 0.5,
        ohem_keep_ratio = cfg.get('segmentation.ohem.keep_ratio', 0.3)
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.get('training.learning_rate', 1e-3),
        momentum=0.9,
        weight_decay=cfg.get('training.weight_decay', 1e-4),
    )
    steps_per_epoch = len(full_loader)

    scheduler       = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps_per_epoch,
        T_mult=1,
        eta_min=cfg.get('training.min_lr', 1e-5),
        last_epoch=-1,
    )
    scaler     = GradScaler()
    warmup_ep  = cfg.get('segmentation.train.warmup_epochs', max(1, cfg.get('training.epochs') // 3))
    seg_thresh = cfg.get('training.threshold', 0.5)
    min_region = cfg.get('postprocessing.min_region', 500)
    cls_weight = cfg.get('segmentation.cls_loss_weight', 1.0)
    best_iou = 0.0
    ssa_iter, full_iter = iter(ssa_loader), iter(full_loader)
    for epoch in range(1, cfg.get('training.epochs') + 1):
        model.train()
        epoch_loss = 0.0
        # three-stage curriculum
        p_full = min(1.0, max(0.0, (epoch - warmup_ep) / warmup_ep))
        for _ in tqdm(range(steps_per_epoch), desc=f"Train Ep{epoch}/{cfg.get('training.epochs')}", ncols=80):
            if torch.rand(1).item() < p_full:
                try:
                    imgs, gts, tissues = next(full_iter)
                except StopIteration:
                    full_iter = iter(full_loader)
                    imgs, gts, tissues = next(full_iter)
            else:
                try:
                    imgs, gts, tissues = next(ssa_iter)
                except StopIteration:
                    ssa_iter = iter(ssa_loader)
                    imgs, gts, tissues = next(ssa_iter)
            imgs, gts, tissues = (
                imgs.to(device, non_blocking=True),
                gts.to(device, non_blocking=True),
                tissues.to(device, non_blocking=True),
            )
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                seg_logits = model(imgs)
                seg_loss   = criterion(seg_logits, gts, tissues)
                # tiny slide-level classifier head
                B,_,H,W  = seg_logits.shape
                flat     = seg_logits.view(B, -1)
                cls_log  = flat.max(dim=1).values
                slide_lb = (gts.view(B, -1).sum(dim=1) > 0).float().to(device)
                cls_loss = F.binary_cross_entropy_with_logits(cls_log, slide_lb)
                loss = seg_loss + cls_weight * cls_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        avg_train = epoch_loss / steps_per_epoch
        print(f"[Seg] Epoch {epoch} Train Loss: {avg_train:.4f}  p_full={p_full:.2f}")
        if logger:
            logger.log_scalar("Train/Loss", avg_train, step=epoch)
            logger.log_scalar("Train/p_full", p_full, step=epoch)
        # ——— VALIDATION ———
        model.eval()
        preds_list, gts_list, tissues_list = [], [], []
        with torch.no_grad():
            for imgs, gts_b, tissues_b in tqdm(val_loader, desc=f"Val   Ep{epoch}/{cfg.get('training.epochs')}", ncols=80):
                imgs   = imgs.to(device, non_blocking=True)
                gts_b  = gts_b.to(device, non_blocking=True)
                tissues_b = tissues_b.to(device, non_blocking=True)
                with autocast(device_type=device.type):
                    logits = model(imgs)
                    probs  = torch.sigmoid(logits)
                # slide‐level gating
                B,_,_,_ = logits.shape
                mvals, _ = probs.view(B, -1).max(dim=1)
                gate = (mvals > seg_thresh).view(B, 1, 1, 1).float()
                probs = probs * gate
                preds_list.append(probs.cpu())
                gts_list.append(gts_b.cpu())
                tissues_list.append(tissues_b.cpu())
        P = torch.cat(preds_list, dim=0)
        G = torch.cat(gts_list, dim=0)
        T = torch.cat(tissues_list, dim=0)
        # only SSA slides
        sums    = G.view(G.size(0), -1).sum(dim=1)
        ssa_idx = (sums > 0).nonzero(as_tuple=False).squeeze(1)
        Pssa, Gssa, Tssa = P[ssa_idx], G[ssa_idx], T[ssa_idx]
        # threshold & remove islands
        bin_pred   = (Pssa > seg_thresh).float() * Tssa
        clean_pred = remove_small_regions_batch(bin_pred, Tssa, min_region)
        inter = (clean_pred * Gssa).view(clean_pred.size(0), -1).sum(dim=1)
        union= ((clean_pred + Gssa) >= 1).view(clean_pred.size(0), -1).sum(dim=1)
        ious  = inter / (union.float() + 1e-8)
        mean_iou = ious.mean().item()
        flat_pr = (clean_pred * Tssa).view(-1)
        flat_gt = (Gssa        * Tssa).view(-1)
        acc     = (flat_pr == flat_gt).float().mean().item()
        print(f"[Seg] Epoch {epoch} Val Acc: {acc:.4f}, IoU: {mean_iou:.4f} @th={seg_thresh:.2f}")
        if logger:
            logger.log_scalar("Eval/Acc", acc, step=epoch)
            logger.log_scalar("Eval/IoU", mean_iou, step=epoch)
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({'model_state': model.state_dict()}, cfg.get('model.saved_path'))
            print(f"[Seg] Saved best IoU: {best_iou:.4f}")
    return model, ssa_loader, val_loader

# -----------------------------------
# Main Pipeline
# -----------------------------------
def main():
    cfg = ConfigManager('config/default_config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = WandbLogger(
        project=cfg.get('logging.project'),
        name=cfg.get('logging.run_name'),
        config=cfg.config
    )
    # 1) Classification pretraining
    #clf_model, _ = pretrain_classifier(cfg, device)
    #all_dataloader = get_mhist_dataloader(
    #    csv_file=cfg.get('data.csv_path'),
    #    img_dir=cfg.get('data.img_dir'),
    #    batch_size=cfg.get('classification.batch_size'),
    #    partition='all',
    #    augmentation_config=cfg.get('augmentation.classification'),
    #    task='classification'
    #)
    #all_dataset = all_dataloader.dataset
    # 2) Generate pseudo-masks
    #if cfg.get('postprocessing.enabled'):
    #    extract_pseudo_masks(clf_model, all_dataset, cfg, device)
    # 3) Train segmentation
    seg_model, _, val_loader = train_segmentation(cfg, device, logger, cfg.get('training.epochs'))
    # 4) Final evaluation & visualization
    print("[Main] Final eval & visualization...")
    evaluate_segmentation(seg_model, val_loader, cfg, device)
    test_loader = get_mhist_dataloader(
        csv_file=cfg.get('data.csv_path'),
        img_dir=cfg.get('data.img_dir'),
        mask_dir=cfg.get('data.mask_dir'),
        batch_size=cfg.get('classification.batch_size'),
        partition='test',
        task='segmentation'
    )
    with torch.no_grad():
        for imgs, msks, _ in test_loader:
            imgs   = imgs.to(device)
            msks   = msks.to(device)
            outs   = seg_model(imgs)
            probs  = torch.sigmoid(outs)
            preds  = (probs > cfg.get('training.threshold')).float()
            visualize_segmentation_results(
                images        = imgs,
                masks         = msks,
                predictions   = preds,
                probabilities = probs,
                step          = cfg.get('training.epochs'),
                logger        = logger
            )
            break

if __name__ == '__main__':
    main()

