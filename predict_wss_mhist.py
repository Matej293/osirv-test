import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchcam.methods import SmoothGradCAMpp
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
import timm
import segmentation_models_pytorch as smp
from datasets.mhist import get_mhist_dataloader
from config.config_manager import ConfigManager
from metrics.wandb_logger import WandbLogger
from utils.visualization import visualize_segmentation_results
import copy
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage.morphology import remove_small_objects
from PIL import Image
from datasets.mhist import MHISTDataset
import torchstain
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

# -----------------------------------
# Loss: Focal Tversky for Classification
# -----------------------------------
class FocalTverskyLoss(nn.Module):
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

# -----------------------------------
# Classifier Pretraining
# -----------------------------------
def pretrain_classifier(config, device):
    """Existing function with modifications"""
    print("[Classifier] Preparing data...")

    # 1) Data loading
    train_base = get_mhist_dataloader(
        csv_file=config.get('data.csv_path'),
        img_dir=config.get('data.img_dir'),
        batch_size=config.get('classification.batch_size'),
        partition='train',
        augmentation_config=config.get('augmentation.classification'),
        task='classification'
    )
    val_loader = get_mhist_dataloader(
        csv_file=config.get('data.csv_path'),
        img_dir=config.get('data.img_dir'),
        batch_size=config.get('classification.batch_size'),
        partition='test',
        augmentation_config=config.get('augmentation.classification'),
        task='classification'
    )
    train_ds = train_base.dataset

    # 2) Weighted sampling for class balance
    labels = train_ds.data['Majority Vote Label'].values
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights), replacement=True)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.get('classification.batch_size'),
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )
    print("[Classifier] Weighted sampling applied")

    # 3) Backbone + deep MLP head
    backbone_name = config.get('classification.backbone')
    print(f"[Classifier] Loading backbone: {backbone_name}")
    if backbone_name in models.__dict__:
        backbone = models.__dict__[backbone_name](weights='DEFAULT')
    else:
        backbone = timm.create_model(backbone_name, pretrained=True)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_feat, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    model = backbone.to(device)

    # 4) Loss definitions
    bce = nn.BCEWithLogitsLoss()
    ft  = FocalTverskyLoss(
        alpha=config.get('classification.tversky_alpha', 0.7),
        beta=config.get('classification.tversky_beta',   0.3),
        gamma=config.get('classification.tversky_gamma',  1.33)
    ).to(device)
    def loss_fn(logits, labels):
        labels_f = labels.float().unsqueeze(1).to(device)
        return bce(logits, labels_f) + ft(logits, labels_f)

    # 5) Training hyperparameters
    base_lr     = config.get('classification.lr')
    mixup_alpha = config.get('classification.mixup_alpha', 0.0)
    best_iou    = 0.0
    epoch_count = 0
    patience = config.get('classification.patience', 10)
    early_stop_counter = 0

    # Keep the same threshold throughout training
    fixed_thresh = config.get('classification.threshold', 0.5)

    # Initialize model state to restore if needed
    best_model_state = None

    # 6) Phase-wise unfreeze based on config
    for phase_idx, phase in enumerate(config.get('classification.unfreeze_schedule')):
        layers = phase['layers']
        epochs = phase['epochs']
        lr_div = phase.get('lr_div', 1)

        if phase_idx > 0:
            # Add warmup period for new layers
            warmup_epochs = 2
            print(f"[Classifier] Starting warmup for newly unfrozen layers ({warmup_epochs} epochs)")
        else:
            warmup_epochs = 0

        # Freeze all then unfreeze head + this phase's layers
        for name, param in model.named_parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith('fc.') or any(layer in name for layer in layers):
                param.requires_grad = True

        # List trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[Classifier] Phase {phase_idx+1}: Trainable parameters: {trainable_params}")

        # Set up optimizer with more conservative LR for deeper layers
        head_params = [p for n,p in model.named_parameters() if n.startswith('fc.') and p.requires_grad]
        layer_params = [p for n,p in model.named_parameters() if any(layer in n for layer in layers) and p.requires_grad]

        # Use much lower LR for backbone layers
        actual_lr_div = lr_div * 2 if phase_idx > 0 else lr_div

        optimizer = optim.AdamW(
            [
                {'params': head_params, 'lr': base_lr},
                {'params': layer_params, 'lr': base_lr / actual_lr_div},
            ],
            weight_decay=config.get('classification.weight_decay')
        )

        # Modify scheduler to have lower peak LR for deeper layers
        max_lr = base_lr if phase_idx == 0 else base_lr / 2

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[max_lr, max_lr / actual_lr_div],  # Different max_lr per param group
            total_steps=epochs * len(train_loader),
            pct_start=0.3 if phase_idx > 0 else 0.1,  # Longer warmup for later phases
            anneal_strategy='linear',
            div_factor=10.0,                         # Initial LR = max_lr/10
            final_div_factor=100.0                   # Final LR = max_lr/1000
        )

        # Warmup phase - very low LR to adapt the unfrozen layers
        if warmup_epochs > 0:
            warmup_optimizer = optim.AdamW(
                [p for n,p in model.named_parameters() if any(layer in n for layer in layers) and p.requires_grad],
                lr=base_lr / (actual_lr_div * 5),  # Even lower LR for warmup
                weight_decay=config.get('classification.weight_decay') * 2  # Higher regularization
            )

            # Just train for a few epochs with very low LR
            for _ in range(warmup_epochs):
                model.train()
                for imgs, labels in tqdm(train_loader, desc=f"Warmup Ep {_+1}/{warmup_epochs}"):
                    imgs, labels = imgs.to(device), labels.to(device)
                    hard_labels = labels.float().unsqueeze(1)

                    warmup_optimizer.zero_grad()
                    logits = model(imgs)
                    loss = loss_fn(logits, labels)
                    loss.backward()
                    # Add gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    warmup_optimizer.step()

            print(f"[Classifier] Warmup complete for phase {phase_idx+1}")

        # Main training loop
        for ep_i in range(epochs):
            epoch_count += 1
            model.train()
            sum_loss = sum_iou = sum_acc = 0.0
            batches = 0

            for imgs, labels in tqdm(train_loader, desc=f"Clf Ep{epoch_count}/{sum(p['epochs'] for p in config.get('classification.unfreeze_schedule'))}"):
                imgs, labels = imgs.to(device), labels.to(device)
                hard_labels  = labels.float().unsqueeze(1)

                # MixUp (affects loss but not metrics)
                if mixup_alpha > 0:
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    idx = torch.randperm(imgs.size(0))
                    imgs   = lam * imgs + (1-lam) * imgs[idx]
                    labels = lam * labels.float() + (1-lam) * labels.float()[idx]

                optimizer.zero_grad()
                logits = model(imgs)
                loss   = loss_fn(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    preds = (torch.sigmoid(logits) > fixed_thresh).float()
                    inter = (preds * hard_labels).sum().item()
                    union = ((preds + hard_labels) >= 1).sum().item()
                    sum_iou += inter / (union + 1e-8)
                    sum_acc += (preds == hard_labels).float().mean().item()

                sum_loss += loss.item()
                batches  += 1

            tr_loss = sum_loss / batches
            tr_iou  = sum_iou  / batches
            tr_acc  = sum_acc  / batches
            print(f"[Classifier] Ep{epoch_count}: Loss={tr_loss:.4f}, IoU={tr_iou:.4f}, Acc={tr_acc:.4f} @th={fixed_thresh:.2f}")

            # Validation
            model.eval()
            val_probs_list, val_labels_list = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    out = model(imgs.to(device))
                    val_probs_list.append(torch.sigmoid(out).cpu())
                    val_labels_list.append(labels.float().unsqueeze(1))
            val_probs = torch.cat(val_probs_list)
            val_labels = torch.cat(val_labels_list)

            # Use fixed threshold - remove threshold sweeping
            val_pred = (val_probs > fixed_thresh).float()
            val_iou = (val_pred * val_labels).sum().item() / (((val_pred + val_labels) >= 1).sum().item() + 1e-8)
            val_acc = (val_pred == val_labels).float().mean().item()

            print(f"[Classifier] Val IoU={val_iou:.4f} @th={fixed_thresh:.2f}, Acc={val_acc:.4f}")

            if val_iou > best_iou:
                best_iou = val_iou
                early_stop_counter = 0
                torch.save(model.state_dict(), config.get('classification.save_path'))
                # Store best model state
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"[Classifier] Saved best IoU: {best_iou:.4f}")
            else:
                early_stop_counter += 1

            # If performance drops too much, restore best model and reduce LR
            if phase_idx > 0 and val_iou < best_iou * 0.8 and ep_i < epochs // 2:
                print(f"[Classifier] Performance dropped too much! Restoring best model and reducing LR")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    # Reduce LR by half
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2.0

            # Early stopping check with patience
            if early_stop_counter >= patience:
                print(f"[Classifier] Early stopping triggered after {epoch_count} epochs")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    # Always load the best model at the end
    try:
        model.load_state_dict(torch.load(config.get('classification.save_path')))
        print(f"[Classifier] Loaded best model with IoU: {best_iou:.4f}")
    except:
        print("[Classifier] Could not load best model, using current model")

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
            cam = cv2.resize(cam, (W,H), interpolation=cv2.INTER_CUBIC)
            cam = (cam - cam.min()) / (cam.ptp() + 1e-8)
        else:
            cam = np.zeros((H, W), dtype=np.float32)

        cam *= tissue_mask

        q    = np.quantile(cam[tissue_mask==1], thresh_q) if label==1 else 1.0
        mask = ((cam > q) & (tissue_mask==1)).astype(np.uint8)
        ko   = np.ones((open_k, open_k), np.uint8)
        kc   = np.ones((close_k, close_k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, ko)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc)
        mask = remove_small_objects(mask.astype(bool), min_size=min_obj).astype(np.uint8)

        # prepare unary for CRF
        prob = np.stack([1-mask, mask], axis=0)
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
        d.addPairwiseBilateral(sxy=bilat_sxy, srgb=bilat_srgb,
                               rgbim=orig, compat=bilat_compat)
        Q = np.array(d.inference(crf_iters))
        refined = Q.argmax(axis=0).reshape(H, W).astype(np.uint8)

        out_name = os.path.splitext(fname)[0] + "_mask.png"
        cv2.imwrite(os.path.join(out_dir, out_name), refined * 255)

    print("[CAM] Pseudo-masks done.")


# -----------------------------------
# Segmentation Training
# -----------------------------------
macenko = torchstain.normalizers.MacenkoNormalizer()

class TorchstainNormalize(A.ImageOnlyTransform):
    """Albumentations wrapper around torchstain.MacenkoNormalizer."""
    def __init__(self, always_apply=False, p=0.7):
        super().__init__(always_apply, p)
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
    
def make_seg_loaders(csv_file, img_dir, mask_dir, batch_size, alb_cfg):
    # 1) Dummy torchvision Resize so MHISTDataset picks up mask_size
    resize_size = tuple(alb_cfg.get("resize", (224, 224)))
    dummy_tf = transforms.Compose([transforms.Resize(resize_size)])

    # 2) Build full train/test MHISTDatasets
    full_train_ds = MHISTDataset(csv_file, img_dir, mask_dir,
                                 transform=dummy_tf,
                                 partition="train",
                                 task="segmentation")
    test_ds       = MHISTDataset(csv_file, img_dir, mask_dir,
                                 transform=dummy_tf,
                                 partition="test",
                                 task="segmentation")

    # 3) Filter SSA-only in the train split
    df = full_train_ds.data  # this is already the "train" partition
    ssa_local_idx = [i for i, lbl in enumerate(df["Majority Vote Label"]) if lbl == 'SSA']
    if len(ssa_local_idx) == 0:
        raise RuntimeError("No SSA samples found in train partition!")
    ssa_train_ds = Subset(full_train_ds, ssa_local_idx)

    # 4) Albumentations pipelines
    train_aug = A.Compose([
        TorchstainNormalize(p=alb_cfg.get("stain_normalize_p", 0.7)),
        A.Resize(*resize_size),
        A.HorizontalFlip(p=alb_cfg.get("h_flip_p", 0.5)),
        A.VerticalFlip(p=alb_cfg.get("v_flip_p", 0.5)),
        A.RandomRotate90(p=alb_cfg.get("rot90_p", 0.5)),
        A.ElasticTransform(
            alpha=alb_cfg.get("elastic_alpha", 1.0),
            sigma=alb_cfg.get("elastic_sigma", 50),
            interpolation=1,  # cv2.INTER_LINEAR
            p=alb_cfg.get("elastic_p", 0.3)
        ),
        A.GridDistortion(p=alb_cfg.get("grid_p", 0.3)),
        A.HueSaturationValue(
            hue_shift_limit=alb_cfg.get("hue", 10),
            sat_shift_limit=alb_cfg.get("sat", 30),
            val_shift_limit=alb_cfg.get("val", 10),
            p=alb_cfg.get("hsv_p", 0.3)
        ),
        A.Normalize(
            mean=alb_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=alb_cfg.get("std",  [0.229, 0.224, 0.225])
        ),
        ToTensorV2(),
    ], additional_targets={"tissue": "mask"})

    val_aug = A.Compose([
        A.Resize(*resize_size),
        A.Normalize(
            mean=alb_cfg.get("mean", [0.485, 0.456, 0.406]),
            std=alb_cfg.get("std",  [0.229, 0.224, 0.225])
        ),
        ToTensorV2(),
    ], additional_targets={"tissue": "mask"})

    # 5) Wrap & apply Albumentations correctly
    class SegWrapper(torch.utils.data.Dataset):
        def __init__(self, base_ds, aug):
            self.ds  = base_ds
            self.aug = aug

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            # 1) Pull back the raw PIL filename from the underlying MHISTDataset
            if isinstance(self.ds, Subset):
                real_idx = self.ds.indices[i]
                mhist_ds = self.ds.dataset
            else:
                real_idx = i
                mhist_ds = self.ds

            # mhist_ds[real_idx] returns (image_tensor, gt_mask, tissue_mask)
            _, gt, tissue = mhist_ds[real_idx]

            # Now re-open the raw PIL to get the original HxW
            row = mhist_ds.data.iloc[real_idx]
            fname = row["Image Name"]
            pil = Image.open(os.path.join(img_dir, fname)).convert("RGB")
            img_np = np.array(pil)                     # H x W x 3

            gt_np     = (gt.squeeze(0).cpu().numpy()*255).astype(np.uint8)     # H x W
            tissue_np = (tissue.squeeze(0).cpu().numpy()*255).astype(np.uint8) # H x W

            augmented = self.aug(image=img_np,
                                 mask=gt_np,
                                 tissue=tissue_np)
            x = augmented["image"]                      # Tensor(3,H,W)
            m = augmented["mask"][None]                 # Tensor(1,H,W)
            t = augmented["tissue"][None]               # Tensor(1,H,W)
            return x, m.float(), t.float()

    train_loader = DataLoader(
        SegWrapper(ssa_train_ds, train_aug),
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        SegWrapper(test_ds, val_aug),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"[SegLoader] train SSA samples: {len(ssa_train_ds)}")
    print(f"[SegLoader] val samples (test partition): {len(test_ds)}")
    return train_loader, val_loader


def evaluate_segmentation(model, val_loader, cfg, device):
    """
    Runs the model over val_loader (which must include both SSA and HP slides) and:
      1. Computes per-image IoU over the full mask (so false-positives on HP hurt you).
      2. Builds a global precision–recall curve over all pixels.
    """
    model.eval()
    all_probs  = []
    all_labels = []
    per_image_ious = []

    with torch.no_grad():
        for imgs, gts, _ in val_loader:
            imgs = imgs.to(device)
            gts  = gts.to(device)           # (B,1,H,W) with 1=SSA, 0=HP/background

            logits = model(imgs)            # (B,1,H,W)
            probs  = torch.sigmoid(logits)  # (B,1,H,W)

            B = probs.size(0)
            # flatten per-image
            for i in range(B):
                p = probs[i,0].cpu().numpy().ravel()
                l = gts[i,0].cpu().numpy().ravel().astype(np.uint8)

                all_probs .append(p)
                all_labels.append(l)

                # per-image IoU at default threshold
                th = cfg.get('training.threshold', 0.5)
                pred_bin = (p > th).astype(np.uint8)
                inter = (pred_bin & l).sum()
                union = ((pred_bin + l) >= 1).sum()
                per_image_ious.append(inter / (union + 1e-6))

    # concatenate for global PR
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # compute PR curve & AUC
    prec, rec, ths = precision_recall_curve(all_labels, all_probs)
    pr_auc = auc(rec, prec)

    # plot
    plt.figure(figsize=(6,6))
    plt.plot(rec, prec, label=f"PR AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Global Precision–Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # report per-image IoU distribution
    ious = np.array(per_image_ious)
    print(f"Per-image IoU @th={cfg.get('training.threshold',0.5):.2f}: "
          f"mean = {ious.mean():.4f}, std = {ious.std():.4f}, n = {len(ious)}")
    print(f"Global PR AUC: {pr_auc:.4f}")

    return {
        "per_image_ious":  ious,
        "precision":      prec,
        "recall":         rec,
        "thresholds":     ths,
        "pr_auc":         pr_auc,
    }

def train_segmentation(cfg, device, logger=None):
    train_loader, val_loader = make_seg_loaders(
        csv_file = cfg.get('data.csv_path'),
        img_dir  = cfg.get('data.img_dir'),
        mask_dir = cfg.get('data.mask_dir'),
        batch_size = cfg.get('data.batch_size'),
        alb_cfg  = {
        'resize': cfg.get('augmentation.segmentation.train.resize'),
        'h_flip_p': cfg.get('augmentation.segmentation.train.horizontal_flip_prob',0.5),
        'v_flip_p': cfg.get('augmentation.segmentation.train.vertical_flip_prob',0.5),
        'hue': cfg.get('augmentation.segmentation.train.hue',0.0),
        'sat': cfg.get('augmentation.segmentation.train.saturation',0.1),
        'val': cfg.get('augmentation.segmentation.train.brightness',1.2)*0,
        'elastic_alpha': cfg.get('augmentation.segmentation.train.elastic_alpha',0.05),
        'elastic_sigma': 50,
        'elastic_affine': 5,
        'grid_p': 0.3,
        'hsv_p': 0.3,
        'mean': [0.485,0.456,0.406],
        'std':  [0.229,0.224,0.225],
        'stain_normalize_p': 0.7
        }
    )

    hp, ssa = 0, 0
    for _, gt, _ in val_loader:
        if gt.sum()>0: ssa+=1
        else:          hp+=1
    print("Val has", ssa, "SSA slides and", hp, "HP slides")

    seg = smp.DeepLabV3Plus(
        encoder_name=cfg.get('model.backbone'),
        encoder_weights='imagenet' if cfg.get('model.pretrained_backbone') else None,
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # masked BCE + Dice
    pos_w = (cfg.get('data.hp_count') + cfg.get('data.ssa_count')) / cfg.get('data.ssa_count')
    def loss_fn(logits, gt, tissue):
        bce_map = nn.BCEWithLogitsLoss(reduction='none',
                                       pos_weight=torch.tensor(pos_w, device=device))(logits, gt)
        bce_val = (bce_map * tissue).sum() / (tissue.sum() + 1e-8)
        probs   = torch.sigmoid(logits)
        inter   = (probs * gt * tissue).sum()
        union   = (probs + gt * tissue).sum()
        dice    = 1 - (2*inter + 1e-5) / (union + 1e-5)
        return bce_val + dice

    opt = optim.AdamW(
        seg.parameters(),
        lr=cfg.get('training.learning_rate'),
        weight_decay=cfg.get('training.weight_decay')
    )
    total_steps = cfg.get('training.epochs') * len(train_loader)
    sched = optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=cfg.get('training.learning_rate'),
        total_steps=total_steps,
        pct_start=0.1
    )

    mixup_alpha = cfg.get('segmentation.mixup_alpha', 0.4)

    best_val_iou = 0.0
    for ep in range(cfg.get('training.epochs')):
        seg.train()
        tot_loss = 0.0

        for x, gt, tissue in tqdm(train_loader, desc=f"Train Ep{ep+1}/{cfg.get('training.epochs')}"):
            x       = x.to(device)
            gt      = gt.to(device)
            tissue  = tissue.to(device)

            # --- MixUp on this batch ---
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(x.size(0))
                x      = lam * x + (1 - lam) * x[idx]
                gt     = lam * gt + (1 - lam) * gt[idx]
                tissue = lam * tissue + (1 - lam) * tissue[idx]

            opt.zero_grad()
            logits = seg(x)

            loss = loss_fn(logits, gt, tissue)
            loss.backward()
            opt.step()
            sched.step()

            tot_loss += loss.item()

        avg_train_loss = tot_loss / len(train_loader)
        print(f"[Seg] Ep{ep+1} Train Loss: {avg_train_loss:.4f}")

        # --- validation & threshold sweep unchanged ---
        seg.eval()
        all_p, all_g, all_t = [], [], []
        with torch.no_grad():
            for x, gt, tissue in val_loader:
                x, gt, tissue = x.to(device), gt.to(device), tissue.to(device)
                p = torch.sigmoid(seg(x))
                all_p.append(p.cpu()); all_g.append(gt.cpu()); all_t.append(tissue.cpu())
        all_p = torch.cat(all_p); all_g = torch.cat(all_g); all_t = torch.cat(all_t)

        best_th, best_iou = 0.5, 0.0
        for th in np.linspace(0.1, 0.9, 17):
            pr = (all_p > th).float() * all_t
            inter = (pr * all_g).sum().item()
            union = ((pr + all_g) >= 1).sum().item()
            iou = inter / (union + 1e-6)
            if iou > best_iou:
                best_iou, best_th = iou, th

        m = all_t.bool()
        acc = ( (all_p>best_th).float()[m] == all_g[m] ).float().mean().item()

        print(f"[Seg] Ep{ep+1} Val Acc: {acc:.4f}, Val IoU: {best_iou:.4f} @th={best_th:.2f}")
        if logger:
            logger.log_scalar('Seg/Val_Acc', acc, step=ep+1)
            logger.log_scalar('Seg/Val_IoU', best_iou, step=ep+1)

        if best_iou > best_val_iou:
            best_val_iou = best_iou
            torch.save({'model_state': seg.state_dict()}, cfg.get('model.saved_path'))
            print(f"[Seg] Saved best IoU: {best_val_iou:.4f}")

    return seg, train_loader

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
    # clf_model, train_ds = pretrain_classifier(cfg, device)

    # load model
    backbone_name = cfg.get('classification.backbone')
    print(f"[Classifier] Loading backbone: {backbone_name}")
    if backbone_name in models.__dict__:
        backbone = models.__dict__[backbone_name](weights='DEFAULT')
    else:
        backbone = timm.create_model(backbone_name, pretrained=True)
    in_feat = backbone.fc.in_features
    backbone.fc = nn.Sequential(
        nn.Linear(in_feat, 2048),
        nn.BatchNorm1d(2048),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(2048, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 1),
    )
    clf_model = backbone.to(device)    
    clf_model.load_state_dict(torch.load(cfg.get('classification.save_path')))
    clf_model.eval()
    clf_model.to(device)

    all_dataloader = get_mhist_dataloader(
        csv_file=cfg.get('data.csv_path'),
        img_dir=cfg.get('data.img_dir'),
        batch_size=cfg.get('classification.batch_size'),
        partition='all',
        augmentation_config=cfg.get('augmentation.classification'),
        task='classification'
    )

    all_dataset = all_dataloader.dataset

    # 2) Generate pseudo-masks
    # if cfg.get('postprocessing.enabled'):
    #    extract_pseudo_masks(clf_model, all_dataset, cfg, device)

    # 3) Train segmentation
    seg_model, train_loader = train_segmentation(cfg, device, logger)

    # 4) Final evaluation & visualization
    print("[Main] Final eval & visualization...")
    seg_model.eval()
    _, val_loader = make_seg_loaders(cfg)
    metrics = evaluate_segmentation(seg_model, val_loader, cfg, device)

    with torch.no_grad():
        for imgs, msks, _ in train_loader:
            imgs = imgs.to(device)
            msks = msks.to(device)
            outs   = seg_model(imgs)
            probs  = torch.sigmoid(outs)
            preds  = (probs > cfg.get('training.threshold')).float()

            visualize_segmentation_results(
                images=imgs,
                masks=msks,
                predictions=preds,
                probabilities=probs,
                step=cfg.get('training.epochs'),
                logger=logger
            )
            break


if __name__ == '__main__':
    main()
