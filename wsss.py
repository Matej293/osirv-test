import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml
import cv2

# Captum for CAM
from captum.attr import LayerGradCam, LayerAttribution
# Deeplab segmentation
from network import modeling
from utils.lovasz_losses import lovasz_hinge

# ----------------------
# Argument Parsing
# ----------------------
parser = argparse.ArgumentParser(description="AffinityNet→MHIST Semi-Supervised Pipeline")
parser.add_argument('--config', type=str, default='config/affinitynet_mhist.yaml', help='Config YAML')
args = parser.parse_args()

# ----------------------
# Load Config
# ----------------------
with open(args.config) as f:
    cfg = yaml.safe_load(f)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Fallback for segmentation CAM seed dir
seg_cam_seeds_dir = cfg['output'].get('seg_cam_seeds', os.path.join(cfg['affinitynet']['seed_dir'], 'seg'))

# ----------------------
# Dataset Loaders
# ----------------------
from datasets.mhist import get_mhist_dataloader

def get_classification_loaders():
    # Build weighted sampler for SSA class
    all_loader = get_mhist_dataloader(
        csv_file=cfg['data']['csv_path'], img_dir=cfg['data']['img_dir'],
        batch_size=1, partition='train', augmentation_config=cfg['augmentation']['train'], task='classification'
    )
    labels = [int(lbl.item()) for _, lbl in all_loader]
    weights = [(cfg['data']['hp_count']/cfg['data']['ssa_count']) if l==1 else 1.0 for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    base = get_mhist_dataloader(
        csv_file=cfg['data']['csv_path'], img_dir=cfg['data']['img_dir'],
        batch_size=cfg['data']['batch_size'], partition='train', augmentation_config=cfg['augmentation']['train'], task='classification'
    )
    train_loader = DataLoader(base.dataset, batch_size=cfg['data']['batch_size'], sampler=sampler)
    val_loader = get_mhist_dataloader(
        csv_file=cfg['data']['csv_path'], img_dir=cfg['data']['img_dir'],
        batch_size=cfg['data']['batch_size'], partition='test', augmentation_config=cfg['augmentation']['test'], task='classification'
    )
    return train_loader, val_loader

# ----------------------
# 1) Train classifier
# ----------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0): super().__init__(); self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0): super().__init__(); self.smooth = smooth
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inter = (inputs * targets).sum()
        denom = inputs.sum() + targets.sum()
        return 1 - (2*inter + self.smooth)/(denom + self.smooth)

def train_classifier():
    train_loader, val_loader = get_classification_loaders()
    # Model
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 1))
    model = model.to(DEVICE)

    focal = FocalLoss(); dice = DiceLoss()
    def combined_loss(logits, labels):
        return focal(logits, labels) + 2.0*dice(logits, labels) + lovasz_hinge(logits, labels)

    total_epochs = sum(p['epochs'] for p in cfg['affinitynet']['unfreeze_schedule'])
    optimizer = optim.AdamW(model.parameters(), lr=cfg['affinitynet']['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    best_val_iou = 0.0
    patience = 0
    max_patience = cfg.get('training', {}).get('patience', 10)
    epoch_idx = 0

    for phase in cfg['affinitynet']['unfreeze_schedule']:
        # Freeze then selectively unfreeze
        for name, param in model.named_parameters(): param.requires_grad = False
        for layer in phase.get('layers', []):
            for name, param in model.named_parameters():
                if layer in name: param.requires_grad = True
        for param in model.fc.parameters(): param.requires_grad = True

        lr = cfg['affinitynet']['lr'] / (2 ** cfg['affinitynet']['unfreeze_schedule'].index(phase))
        optimizer.param_groups[0]['lr'] = lr

        for _ in range(phase['epochs']):
            epoch_idx += 1
            # Training loop
            model.train()
            total_loss = 0.0
            inter = union = corr = total = 0
            for imgs, labels in tqdm(train_loader, desc=f"Clf Train Epoch {epoch_idx}/{total_epochs}"):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
                optimizer.zero_grad()
                logits = model(imgs)
                loss = combined_loss(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                corr += (preds == labels).sum().item(); total += labels.numel()
                inter += (preds*labels).sum().item(); union += ((preds+labels)>=1).sum().item()
            scheduler.step()
            train_iou = inter/(union+1e-8); train_acc = corr/total
            print(f"Clf Epoch {epoch_idx} Train IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")

            # Validation + bias calibration
            model.eval()
            logits_list, labels_list = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    logits_list.append(model(imgs.to(DEVICE)).cpu())
                    labels_list.append(labels.unsqueeze(1).float())
            logits_all = torch.cat(logits_list, dim=0)
            labels_all = torch.cat(labels_list, dim=0)

            # Learnable bias
            bias = nn.Parameter(torch.zeros(1, device=DEVICE))
            opt_b = optim.LBFGS([bias], lr=0.1)
            def calibrate():
                opt_b.zero_grad()
                preds = torch.sigmoid(logits_all.to(DEVICE) + bias)
                l = F.binary_cross_entropy(preds, labels_all.to(DEVICE))
                l.backward()
                return l
            for _ in range(20): opt_b.step(calibrate)

            with torch.no_grad():
                preds = (torch.sigmoid(logits_all.to(DEVICE) + bias) > 0.5).float()
                inter_v = (preds * labels_all.to(DEVICE)).sum().item()
                union_v = ((preds + labels_all.to(DEVICE)) >= 1).sum().item()
                val_iou = inter_v/(union_v+1e-8)
                val_acc = (preds == labels_all.to(DEVICE)).sum().item()/labels_all.numel()
            print(f"Clf Epoch {epoch_idx} Val IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")

            if val_iou > best_val_iou:
                best_val_iou = val_iou; patience = 0
                torch.save(model.state_dict(), cfg['affinitynet']['cls_weights'])
                print(f"Saved classifier @IoU={val_iou:.4f}")
            else:
                patience += 1
                if patience >= max_patience:
                    print("Early stopping classifier on IoU")
                    return model
    return model

# ----------------------
# 2) Generate CAM seeds
# ----------------------
def generate_cam_seeds(model, layer_name, seed_dir):
    layer = getattr(model, layer_name)[-1]
    gc = LayerGradCam(model, layer)
    os.makedirs(seed_dir, exist_ok=True)
    for img in tqdm(os.listdir(cfg['data']['img_dir']), desc="Gen CAM Seeds"):
        if not img.lower().endswith('.png'): continue
        im = Image.open(os.path.join(cfg['data']['img_dir'], img)).convert('RGB')
        tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        inp = tf(im).unsqueeze(0).to(DEVICE)
        attr = gc.attribute(inp, target=0)
        up = LayerAttribution.interpolate(attr, inp.shape[2:]).squeeze().cpu().detach().numpy()
        heat = (np.clip(up, 0, None) - up.min())/(up.max() - up.min() + 1e-8)
        thr = np.percentile(heat, cfg['affinitynet'].get('seed_percentile', 50))
        seed = (heat >= thr).astype(np.uint8) * 255
        seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        Image.fromarray(seed).save(os.path.join(seed_dir, img.replace('.png','_seed.png')))

# ----------------------
# 3) Segmentation training
# ----------------------
def train_segmentation(pseudo_dir):
    from datasets.mhist import MHISTDataset
    seg_tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    ds = MHISTDataset(csv_file=cfg['data']['csv_path'], img_dir=cfg['data']['img_dir'],
                      mask_dir=pseudo_dir, transform=seg_tf, partition='train', task='segmentation')
    dl = DataLoader(ds, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=4)
    model = modeling.deeplabv3plus_resnet101(
        num_classes=1, output_stride=cfg['model']['output_stride'], pretrained_backbone=True
    ).to(DEVICE)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    for ep in range(cfg['training']['epochs']):
        model.train(); tot_loss=0; count=0
        for imgs, masks in tqdm(dl, desc=f"Seg Train {ep+1}/{cfg['training']['epochs']}"):
            imgs, masks = imgs.to(DEVICE), masks.unsqueeze(1).to(DEVICE).float()
            opt.zero_grad(); out = model(imgs); loss = crit(out, masks)
            loss.backward(); opt.step(); tot_loss+=loss.item(); count+=1
        print(f"Seg Epoch {ep+1} Loss {tot_loss/count:.4f}")
    return model

# ----------------------
# 4) CAM from segmentation
# ----------------------
def generate_seg_cam(model_seg, seed_dir):
    """
    Run Grad-CAM on the segmentation network to produce new CAM seeds.
    Assumes model_seg is a DeepLabV3+ instance.
    """
    # 1) Put model into eval mode so BatchNorm won't break on batch size=1
    model_seg.eval()

    # 2) Create our LayerGradCam on the last backbone layer
    layer = model_seg.backbone.layer4[-1]
    gc    = LayerGradCam(model_seg, layer)

    os.makedirs(seed_dir, exist_ok=True)

    # 3) Define the image transform
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    # 4) Loop over all inputs
    for img_name in tqdm(os.listdir(cfg['data']['img_dir']), desc="Gen Seg CAM"):
        if not img_name.lower().endswith('.png'):
            continue

        img = Image.open(os.path.join(cfg['data']['img_dir'], img_name)).convert('RGB')
        inp = tf(img).unsqueeze(0).to(DEVICE)

        # 5) Compute Grad-CAM in no-grad mode
        with torch.no_grad():
            attr = gc.attribute(inp, target=0)

        # 6) Upsample and normalize
        up    = LayerAttribution.interpolate(attr, inp.shape[2:])
        up_np = up.squeeze().cpu().numpy()
        heat  = np.clip(up_np, 0, None)
        heat  = (heat - heat.min())/(heat.max() - heat.min() + 1e-8)

        # 7) Threshold at your configured percentile
        pct   = cfg['affinitynet'].get('seed_percentile', 50)
        thr   = np.percentile(heat, pct)
        mask  = (heat >= thr).astype(np.uint8)

        # 8) Save to *_seg_seed.png
        out_path = os.path.join(seed_dir, img_name.replace('.png','_seg_seed.png'))
        Image.fromarray(mask * 255).save(out_path)


# ----------------------
# 5) Affinity refinement & final masks
# ----------------------
def make_feature_extractor():
    res = models.resnet50(weights=None)
    return nn.Sequential(res.conv1, res.bn1, res.relu, res.maxpool,
                         res.layer1, res.layer2, res.layer3, res.layer4)
class AffinityNet(nn.Module):
    def __init__(self, feat_extractor): super().__init__(); self.feat_extractor=feat_extractor; self.refine = nn.Conv2d(2048, 2, kernel_size=1)
    def forward(self, x): return self.refine(self.feat_extractor(x))

def train_affinity(feat_extractor, seed_dir):
    """
    Train AffinityNet refinement head on segmentation‐CAM seeds.

    Args:
        feat_extractor (nn.Module): backbone up through resnet50.layer4
        seed_dir (str): directory containing *_seg_seed.png files

    Returns:
        feat_extractor, trained_affinity_model
    """
    import cv2
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms

    # 1) CAM‐seed dataset with explicit negative channel
    class CAMSeedDataset(Dataset):
        def __init__(self, img_dir, seed_dir, transform, neg_thresh):
            self.imgs      = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            self.img_dir   = img_dir
            self.seed_dir  = seed_dir
            self.transform = transform
            self.neg_thresh = neg_thresh

        def __len__(self):
            return len(self.imgs)

        def __getitem__(self, idx):
            name = self.imgs[idx]
            img  = Image.open(os.path.join(self.img_dir, name)).convert('RGB')
            seed_path = os.path.join(self.seed_dir, name.replace('.png', '_seed.png'))
            seed_img  = Image.open(seed_path).convert('L')
            seed_np   = np.array(seed_img) / 255.0

            # explicit negative mask
            neg_mask = (seed_np < cfg['affinitynet']['seed_neg_thresh']).astype(np.float32)
            pos_mask = seed_np.astype(np.float32)
            stacked  = np.stack([neg_mask, pos_mask], axis=0)  # [2, H, W]

            return self.transform(img), torch.tensor(stacked, dtype=torch.float32)

    # 2) DataLoader
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])
    ds     = CAMSeedDataset(cfg['data']['img_dir'],
                            seed_dir,
                            tf,
                            neg_thresh=cfg['affinitynet']['seed_neg_thresh'])
    loader = DataLoader(ds,
                        batch_size=cfg['affinitynet']['batch_size'],
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

    # 3) AffinityNet head: 2‐channel output
    class AffinityNetHead(nn.Module):
        def __init__(self, feat_extractor):
            super().__init__()
            self.feat_extractor = feat_extractor
            self.refine        = nn.Conv2d(2048, 2, kernel_size=1)

        def forward(self, x):
            feats = self.feat_extractor(x)
            return self.refine(feats)

    model = AffinityNetHead(feat_extractor).to(DEVICE)

    # 4) Losses, optimizer, scheduler
    bce_loss   = nn.BCEWithLogitsLoss()
    dice_loss  = DiceLoss()
    lr         = cfg['affinitynet']['aff_lr'] * 5  # stronger initial LR
    optimizer  = optim.AdamW(model.refine.parameters(), lr=lr, weight_decay=0.01)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                     optimizer,
                     T_max=cfg['affinitynet']['aff_epochs'] + 5
                 )

    # 5) Training loop
    for ep in range(cfg['affinitynet']['aff_epochs'] + 5):
        model.train()
        total_loss = 0.0
        pix_corr   = pix_total = inter = uni = 0

        for imgs, seeds in tqdm(loader,
                                desc=f"Affinity Epoch {ep+1}/{cfg['affinitynet']['aff_epochs']+5}"):
            imgs = imgs.to(DEVICE)                            # [B,3,H,W]
            seeds = seeds.to(DEVICE)                          # [B,2,H,W]

            # extract features (no grads)
            with torch.no_grad():
                feats = feat_extractor(imgs)                  # [B,2048,Hf,Wf]
            _, _, Hf, Wf = feats.shape

            # downsample seeds to feature map size
            seeds_ds = F.adaptive_avg_pool2d(seeds, (Hf, Wf))  # [B,2,Hf,Wf]

            # forward pass
            logits = model(imgs)                              # [B,2,Hf,Wf]

            # compute losses
            loss_bce = bce_loss(logits, seeds_ds)
            loss_d   = dice_loss(logits[:,1:2], seeds_ds[:,1:2])
            loss     = loss_bce + cfg['affinitynet']['loss_dice_weight'] * loss_d

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # metrics: pixel‐acc and IoU on fg channel
            probs = torch.sigmoid(logits[:,1:2])
            preds = (probs > 0.5).float()
            gt    = seeds_ds[:,1:2]

            pix_corr += (preds == gt).sum().item()
            pix_total+= gt.numel()
            inter    += (preds * gt).sum().item()
            uni      += ((preds + gt) >= 1).sum().item()

        scheduler.step()

        avg_loss = total_loss / len(loader)
        pix_acc  = pix_corr / pix_total
        pix_iou  = inter / (uni + 1e-8)

        print(f"Affinity Epoch {ep+1} — "
              f"Loss: {avg_loss:.4f} | "
              f"PixAcc: {pix_acc:.4f} | "
              f"IoU: {pix_iou:.4f}")

    return feat_extractor, model


def make_final_masks(feat_extractor, affnet, mask_dir):
    os.makedirs(mask_dir, exist_ok=True)
    tf = transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    for img in tqdm(os.listdir(cfg['data']['img_dir']), desc="Final Masks"):
        if not img.endswith('.png'): continue
        inp = tf(Image.open(os.path.join(cfg['data']['img_dir'], img)).convert('RGB')).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): logits = affnet(inp)
        p= torch.sigmoid(logits)[0,0].cpu().numpy()
        mask = (p >= np.percentile(p, cfg['affinitynet'].get('prop_percentile',50))).astype(np.uint8)
        Image.fromarray(mask*255).save(os.path.join(mask_dir, img))

# ----------------------
# Main function
# ----------------------
if __name__ == '__main__':
    # 1) Train the image-level classifier
    #cls_model = train_classifier()

    # 2) Generate CAM seeds from the classifier
    # generate_cam_seeds(
    #     model=cls_model,
    #     layer_name='layer4',
    #     seed_dir=cfg['affinitynet']['seed_dir']
    # )

    # # 3) AffinityNet refinement on those classifier seeds
    # feat_extractor = make_feature_extractor().to(DEVICE)
    # feat_extractor, aff_model = train_affinity(
    #     feat_extractor=feat_extractor,
    #     seed_dir=cfg['affinitynet']['seed_dir']
    # )

    # # 4) Produce the first round of pseudo-masks
    # make_final_masks(
    #     feat_extractor=feat_extractor,
    #     affnet=aff_model,
    #     mask_dir=cfg['output']['pseudo_dir']
    # )

    # 5) Train DeepLabV3+ on those pseudo-masks
    seg_model = train_segmentation(
        pseudo_dir=cfg['output']['pseudo_dir']
    )

    # 6) Re-extract CAM seeds from the new segmentation model
    generate_seg_cam(
        model_seg=seg_model,
        seed_dir=seg_cam_seeds_dir
    )

    # 7) A second AffinityNet pass on the segmentation-CAM seeds
    feat_extractor2 = make_feature_extractor().to(DEVICE)
    feat_extractor2, aff_model2 = train_affinity(
        feat_extractor=feat_extractor2,
        seed_dir=seg_cam_seeds_dir
    )

    # 8) Final pseudo-mask generation
    make_final_masks(
        feat_extractor=feat_extractor2,
        affnet=aff_model2,
        mask_dir=cfg['output']['pseudo_dir']
    )

    print("End-to-end semi-supervised pipeline complete.")

