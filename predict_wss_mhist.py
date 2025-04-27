import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torchvision import models
from torchcam.methods import SmoothGradCAMpp
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax
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
from sklearn.metrics import precision_recall_curve, auc
from utils.utils import *

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
    ft  = ClassifierFocalTverskyLoss(
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
                    # reducing LR by half
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2.0

            # early stopping
            if early_stop_counter >= patience:
                print(f"[Classifier] Early stopping triggered after {epoch_count} epochs")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

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
def make_seg_loaders(cfg):
    resize = tuple(cfg.get('augmentation.segmentation.train.resize', (224,224)))
    # 1) dummy transform so MHISTDataset knows mask_size
    dummy_tf = transforms.Compose([transforms.Resize(resize)])
    full_train = MHISTDataset(
        cfg.get('data.csv_path'),
        cfg.get('data.img_dir'),
        mask_dir=cfg.get('data.mask_dir'),
        transform=dummy_tf,
        partition='train',
        task='segmentation'
    )
    test_ds = MHISTDataset(
        cfg.get('data.csv_path'),
        cfg.get('data.img_dir'),
        mask_dir=cfg.get('data.mask_dir'),
        transform=transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(
                cfg.get('augmentation.segmentation.test.mean'),
                cfg.get('augmentation.segmentation.test.std')
            )
        ]),
        partition='test',
        task='segmentation'
    )

    # 2) SSA-only subset
    ssa_idxs = [i for i, lbl in enumerate(full_train.data["Majority Vote Label"]) if lbl == 'SSA']
    if not ssa_idxs:
        raise RuntimeError("No SSA samples in train partition!")
    ssa_train = Subset(full_train, ssa_idxs)
    print(f"[SegLoader] train SSA={len(ssa_train)}  val={len(test_ds)}")

    # 3) Albumentations for train
    alb = cfg.get('augmentation.segmentation.train')
    train_aug = A.Compose([
        TorchstainNormalize(p=alb.get('stain_normalize_p',0.7)),
        A.Resize(*resize),
        A.HorizontalFlip(p=alb.get('horizontal_flip_prob',0.5)),
        A.VerticalFlip(p=alb.get('vertical_flip_prob',0.5)),
        A.RandomRotate90(p=alb.get('random_rotate90_prob',0.5)),
        A.ElasticTransform(p=alb.get('elastic_p',0.3),
                           alpha=alb.get('elastic_alpha',1.0),
                           sigma=alb.get('elastic_sigma',50)),
        A.GridDistortion(p=alb.get('grid_distortion_prob',0.3)),
        A.HueSaturationValue(p=alb.get('hsv_prob',0.3),
                             hue_shift_limit=alb.get('hue_shift_limit',10),
                             sat_shift_limit=alb.get('sat_shift_limit',30),
                             val_shift_limit=alb.get('val_shift_limit',10)),
        A.Normalize(mean=alb.get('mean'), std=alb.get('std')),
        ToTensorV2(),
    ], additional_targets={'tissue':'mask'})

    class TrainWrapper(torch.utils.data.Dataset):
        def __init__(self, base_ds, aug, img_dir):
            self.base = base_ds
            self.aug  = aug
            self.img_dir = img_dir

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            # resolve real idx and raw MHISTDataset
            if isinstance(self.base, Subset):
                real_idx = self.base.indices[i]
                raw_ds   = self.base.dataset
            else:
                real_idx = i
                raw_ds   = self.base
            # a) reload raw PIL for correct HxW
            row   = raw_ds.data.iloc[real_idx]
            fname = row["Image Name"]
            pil   = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
            img_np = np.array(pil)  # HxWx3 uint8

            # b) get masks (already resized via dummy_tf)
            _, gt_mask, tissue_mask = raw_ds[real_idx]
            gt_np     = (gt_mask.squeeze(0).numpy()*255).astype(np.uint8)
            tissue_np = (tissue_mask.squeeze(0).numpy()*255).astype(np.uint8)

            # c) apply Albumentations
            d = self.aug(image=img_np, mask=gt_np, tissue=tissue_np)
            x = d['image']
            m = d['mask'][None].float()/255.0
            t = d['tissue'][None].float()/255.0
            return x, m, t

    train_loader = DataLoader(
        TrainWrapper(ssa_train, train_aug, cfg.get('data.img_dir')),
        batch_size=cfg.get('data.batch_size'), shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        test_ds,
        batch_size=cfg.get('data.batch_size'),
        shuffle=False, num_workers=4, pin_memory=True
    )
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
        "pred_flat":      all_probs,
        "targets_flat":   all_labels,
    }

def train_segmentation(cfg, device, logger=None, step=None):
    train_loader, val_loader = make_seg_loaders(cfg)

    seg = smp.DeepLabV3Plus(
        encoder_name=cfg.get('model.backbone'),
        encoder_weights='imagenet' if cfg.get('model.pretrained_backbone') else None,
        in_channels=3, classes=1, activation=None
    ).to(device)

    pos_w = (cfg.get('data.hp_count')+cfg.get('data.ssa_count'))/cfg.get('data.ssa_count')
    bce_map = nn.BCEWithLogitsLoss(
        reduction='none',
        pos_weight=torch.tensor(pos_w,device=device)
    )
    def loss_fn(logits, gt, tissue):
        m = tissue
        b = (bce_map(logits,gt)*m).sum()/(m.sum()+1e-8)
        p = torch.sigmoid(logits)
        inter = (p*gt*m).sum()
        union = ((p+gt)*m).sum()
        dice = 1 - (2*inter+1e-5)/(union+1e-5)
        return b + dice

    optimizer = optim.AdamW(seg.parameters(),
                            lr=cfg.get('training.learning_rate'),
                            weight_decay=cfg.get('training.weight_decay'))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.get('training.learning_rate'),
        total_steps=cfg.get('training.epochs')*len(train_loader),
        pct_start=0.1
    )

    best_iou = 0.0
    for ep in range(cfg.get('training.epochs')):
        seg.train(); tr_loss=0.0
        for x,gt,t in tqdm(train_loader, desc=f"Train Ep{ep+1}/{cfg.get('training.epochs')}"):
            x,gt,t = x.to(device),gt.to(device),t.to(device)
            optimizer.zero_grad()
            l = loss_fn(seg(x),gt,t)
            l.backward(); optimizer.step(); scheduler.step()
            tr_loss += l.item()
        print(f"[Seg] Ep{ep+1} Train Loss: {tr_loss/len(train_loader):.4f}")

        # — fast, in-RAM val —
        seg.eval()
        Ps, Gs, Ts = [],[],[]
        with torch.no_grad():
            for x,gt,t in val_loader:
                x,gt,t = x.to(device),gt.to(device),t.to(device)
                Ps.append(torch.sigmoid(seg(x)).cpu())
                Gs.append(gt.cpu()); Ts.append(t.cpu())
        P = torch.cat(Ps); G = torch.cat(Gs); T = torch.cat(Ts)
        N,M,H,W = P.shape

        mask_ssa = (G.view(N,-1).sum(dim=1)>0)
        Pssa = P[mask_ssa]; Gssa = G[mask_ssa]; Tssa = T[mask_ssa]
        Pf = (Pssa>0.5).float()*Tssa

        # per-image IoU
        inter = (Pf*Gssa).view(Pssa.size(0),-1).sum(dim=1)
        union = (((Pf+Gssa)>=1).view(Pssa.size(0),-1)).sum(dim=1)
        mean_iou = (inter/(union+1e-8)).mean().item()

        flat_pr = Pf.view(-1); flat_gt = Gssa.view(-1)
        acc = (flat_pr==flat_gt).float().mean().item()

        print(f"[Seg] Ep{ep+1} Val Acc: {acc:.4f}, IoU: {mean_iou:.4f} @th=0.50")

        if mean_iou>best_iou:
            best_iou=mean_iou
            torch.save({'model_state':seg.state_dict()},
                       cfg.get('model.saved_path'))
            print(f"[Seg] Saved best IoU: {best_iou:.4f}")

    return seg, train_loader, val_loader


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

    # all_dataloader = get_mhist_dataloader(
    #     csv_file=cfg.get('data.csv_path'),
    #     img_dir=cfg.get('data.img_dir'),
    #     batch_size=cfg.get('classification.batch_size'),
    #     partition='all',
    #     augmentation_config=cfg.get('augmentation.classification'),
    #     task='classification'
    # )

    # all_dataset = all_dataloader.dataset

    # 2) Generate pseudo-masks
    # if cfg.get('postprocessing.enabled'):
    #    extract_pseudo_masks(clf_model, all_dataset, cfg, device)

    # 3) Train segmentation
    seg_model, _, val_loader = train_segmentation(cfg, device, logger, cfg.get('training.epochs'))

    # 4) Final evaluation & visualization
    print("[Main] Final eval & visualization...")
    eval_dict = evaluate_segmentation(seg_model, val_loader, cfg, device)

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
                images      = imgs,
                masks       = msks,
                predictions = preds,
                probabilities = probs,
                step        = cfg.get('training.epochs'),
                logger      = logger
            )
            break

if __name__ == '__main__':
    main()
