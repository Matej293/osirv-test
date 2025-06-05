import os
from pathlib import Path
from typing import Optional, Tuple, Any, List, Dict, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import cv2

import albumentations as A
from albumentations.pytorch import ToTensorV2

from utils.utils import TorchstainNormalize, remove_small_regions_batch


# =============================================================================
# Helper functions
# =============================================================================

def _get_mask_filename(image_filename: str) -> str:
    return image_filename.replace(".png", "_mask.png")


def _load_rgb_np(img_dir: str, filename: str) -> Tuple[np.ndarray, Image.Image]:
    """
    Loads an RGB image from disk and returns (numpy array, PIL Image).
    """
    path = os.path.join(img_dir, filename)
    pil_img = Image.open(path).convert("RGB")
    img_np = np.array(pil_img)
    return img_np, pil_img


def _compute_tissue_mask(img_np: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale and threshold to identify tissue vs. background.
    Returns a full‐resolution uint8 mask of shape (H, W), values 0 or 255.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) / 255.0
    tissue_binary = (gray < 0.95).astype(np.uint8) * 255
    return tissue_binary.astype(np.uint8)


def _load_full_gt_mask(mask_dir: str, mask_filename: str) -> np.ndarray:
    """
    Loads the ground‐truth segmentation mask (full resolution) from disk,
    binarizes (>50 → 255), and returns a uint8 array of shape (H, W).
    """
    path = os.path.join(mask_dir, mask_filename)
    pil_mask = Image.open(path).convert("L")
    mask_arr = np.array(pil_mask)
    binary = (mask_arr > 50).astype(np.uint8) * 255
    return binary.astype(np.uint8)


# =============================================================================
# MHISTDataset
# =============================================================================

class MHISTDataset(Dataset):
    """
    MHIST dataset for classification or segmentation.
    - Classification: returns (image_tensor, label_tensor).
    - Segmentation: returns (image_tensor, gt_mask_tensor, tissue_mask_tensor).

    When `task="segmentation"`, the provided `transform` must include a Resize so
    that all outputs end up at a consistent size.
    """
    def __init__(
        self,
        csv_file: str,
        img_dir: str,
        mask_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        partition: str = "train",
        task: str = "classification"
    ) -> None:
        csv_path = Path(csv_file)
        img_path = Path(img_dir)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        if not img_path.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if task == "segmentation":
            if mask_dir is None or not Path(mask_dir).exists():
                raise FileNotFoundError("Mask directory required for segmentation")

        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.task = task

        # Filter by partition
        if partition in ("train", "test"):
            self.data = self.data[self.data["Partition"] == partition]
        elif partition != "all":
            raise ValueError("partition must be 'train', 'test', or 'all'")
        if self.data.empty:
            raise ValueError(f"No samples in partition='{partition}'")

        # Map labels "HP"→0, "SSA"→1
        self.data["label"] = self.data["Majority Vote Label"].map({"HP": 0, "SSA": 1})

        if self.task == "segmentation":
            # We will rely on Albumentations Resize to standardize to (224,224), so no pre‐set here.
            self.mask_size = (224, 224)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        row = self.data.iloc[idx]
        filename = row["Image Name"]

        # 1) Load RGB full‐resolution
        img_np, img_pil = _load_rgb_np(self.img_dir, filename)
        tissue_full = _compute_tissue_mask(img_np)  # full‐res uint8 mask

        # 2) Transform the PIL image if requested
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        if self.task == "classification":
            label_tensor = torch.tensor(row["label"], dtype=torch.long)
            return img_tensor, label_tensor

        # 3) Segmentation branch: load full‐res GT mask, full‐res tissue mask
        mask_filename = _get_mask_filename(filename)
        gt_full = _load_full_gt_mask(self.mask_dir, mask_filename)       # shape (H, W), {0,255}
        tissue_full_uint8 = tissue_full                                    # shape (H, W), {0,255}

        # Convert both to float tensors in [0,1], but expansion to (1,H,W) done in wrapper
        gt_mask_tensor = torch.from_numpy((gt_full.astype(np.float32) / 255.0))[None, ...]       # (1, H, W)
        tissue_mask_tensor = torch.from_numpy((tissue_full_uint8.astype(np.float32) / 255.0))[None, ...]  # (1, H, W)

        return img_tensor, gt_mask_tensor, tissue_mask_tensor


# =============================================================================
# Segmentation Wrappers
# =============================================================================

class SegBaseWrapper(Dataset):
    """
    Base wrapper for segmentation: reopens raw samples, computes tissue/mask,
    then delegates augmentation to child classes.
    """
    def __init__(
        self,
        base_dataset: MHISTDataset,
        indices: List[int],
        img_dir: str,
        mask_dir: str,
        resize: Tuple[int, int]
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.resize = resize  # (height, width)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        real_idx = self.indices[idx]
        raw_ds = self.base_dataset
        row = raw_ds.data.iloc[real_idx]
        filename = row["Image Name"]

        # 1) Load raw image & compute full‐res tissue
        img_np, _ = _load_rgb_np(self.img_dir, filename)
        tissue_full = _compute_tissue_mask(img_np)  # full‐res uint8 mask

        # 2) Load the full‐res GT mask
        mask_filename = _get_mask_filename(filename)
        gt_full = _load_full_gt_mask(self.mask_dir, mask_filename)  # (H_full, W_full) uint8

        # 3) Delegate to augmentation routine in subclass
        return self.apply_aug(img_np, gt_full, tissue_full)

    def apply_aug(
        self,
        img_np: np.ndarray,
        gt_mask_uint8: np.ndarray,
        tissue_uint8: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError


class SegTrainWrapper(SegBaseWrapper):
    """
    Segmentation wrapper for *training*:
    - Resize the full‐resolution image/mask/tissue to (resize_h, resize_w)
    - (Optional) stain normalize
    - Random flips / elastic / grid / color jitter
    - Normalize & ToTensorV2
    """
    def __init__(
        self,
        base_dataset: MHISTDataset,
        indices: List[int],
        img_dir: str,
        mask_dir: str,
        alb_cfg: Dict[str, Any],
        resize: Tuple[int, int]
    ) -> None:
        super().__init__(base_dataset, indices, img_dir, mask_dir, resize)

        resize_h, resize_w = resize

        self.aug = A.Compose(
            [
                # 1) Optional stain normalization (on PIL→NumPy)
                TorchstainNormalize(p=alb_cfg.get("stain_normalize_p", 0.7)),
                # 2) Resize to (resize_h, resize_w)
                A.Resize(resize_h, resize_w),
                # 3) Random flips / rotations / elastic / grid / color jitter
                A.HorizontalFlip(p=alb_cfg.get("h_flip_p", 0.5)),
                A.VerticalFlip(p=alb_cfg.get("v_flip_p", 0.5)),
                A.RandomRotate90(p=alb_cfg.get("rot90_p", 0.5)),
                A.ElasticTransform(
                    p=alb_cfg.get("elastic_p", 0.3),
                    alpha=alb_cfg.get("elastic_alpha", 1.0),
                    sigma=alb_cfg.get("elastic_sigma", 50),
                ),
                A.GridDistortion(p=alb_cfg.get("grid_p", 0.3)),
                A.HueSaturationValue(
                    p=alb_cfg.get("hsv_p", 0.3),
                    hue_shift_limit=alb_cfg.get("hue", 10),
                    sat_shift_limit=alb_cfg.get("sat", 30),
                    val_shift_limit=alb_cfg.get("val", 10),
                ),
                # 4) Final normalize & to‐tensor
                A.Normalize(
                    mean=alb_cfg.get("mean", [0.485, 0.456, 0.406]),
                    std=alb_cfg.get("std", [0.229, 0.224, 0.225]),
                ),
                ToTensorV2(),
            ],
            additional_targets={"tissue": "mask"},
        )

    def apply_aug(
        self,
        img_np: np.ndarray,
        gt_mask_uint8: np.ndarray,
        tissue_uint8: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        `img_np`, `gt_mask_uint8`, `tissue_uint8` are full resolution (H_full, W_full).
        Albumentations pipeline does:
          - TorchstainNormalize
          - Resize to (resize_h, resize_w)
          - Flips / distortions
          - Normalize + ToTensorV2
        """
        augmented = self.aug(
            image=img_np,
            mask=gt_mask_uint8,
            tissue=tissue_uint8
        )
        img_tensor    = augmented["image"]                    # (3, resize_h, resize_w)
        mask_tensor   = augmented["mask"].unsqueeze(0).float() / 255.0   # (1, resize_h, resize_w)
        tissue_tensor = augmented["tissue"].unsqueeze(0).float() / 255.0 # (1, resize_h, resize_w)
        return img_tensor, mask_tensor, tissue_tensor


class SegValWrapper(SegBaseWrapper):
    """
    Segmentation wrapper for *validation*:
    - Resize full‐res → (resize_h, resize_w)
    - Normalize + ToTensorV2
    """
    def __init__(
        self,
        base_dataset: MHISTDataset,
        indices: List[int],
        img_dir: str,
        mask_dir: str,
        val_cfg: Dict[str, Any],
        resize: Tuple[int, int]
    ) -> None:
        super().__init__(base_dataset, indices, img_dir, mask_dir, resize)

        resize_h, resize_w = resize
        self.aug = A.Compose(
            [
                A.Resize(resize_h, resize_w),
                A.Normalize(
                    mean=val_cfg.get("mean", [0.485, 0.456, 0.406]),
                    std=val_cfg.get("std", [0.229, 0.224, 0.225]),
                ),
                ToTensorV2(),
            ],
            additional_targets={"tissue": "mask"},
        )

    def apply_aug(
        self,
        img_np: np.ndarray,
        gt_mask_uint8: np.ndarray,
        tissue_uint8: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        `img_np`, `gt_mask_uint8`, `tissue_uint8` are full resolution.
        Albumentations pipeline will:
          - Resize all to (resize_h, resize_w)
          - Normalize + ToTensorV2
        """
        augmented = self.aug(
            image=img_np,
            mask=gt_mask_uint8,
            tissue=tissue_uint8
        )
        img_tensor    = augmented["image"]                    # (3, resize_h, resize_w)
        mask_tensor   = augmented["mask"].unsqueeze(0).float() / 255.0   # (1, resize_h, resize_w)
        tissue_tensor = augmented["tissue"].unsqueeze(0).float() / 255.0 # (1, resize_h, resize_w)
        return img_tensor, mask_tensor, tissue_tensor


# =============================================================================
# Combined Loader Function
# =============================================================================

def get_mhist_loaders(
    cfg: Dict[str, Any]
) -> Union[DataLoader, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Returns:
      - If task="classification": (train_loader, val_loader).
      - If task="segmentation": (pos_loader, neg_loader, val_loader).

    Expected cfg keys:
      - task: "classification" or "segmentation"
      - data.csv_path: path to CSV
      - data.img_dir
      - data.mask_dir (for segmentation)
      - data.batch_size
      - data.num_workers (optional)
      - augmentation.classification (dict) [for classification]
      - augmentation.segmentation.train (dict) [for segmentation]
      - augmentation.segmentation.test (dict) [for segmentation]
    """
    task        = cfg.get("task", "classification")
    csv_file    = cfg.get("data.csv_path")
    img_dir     = cfg.get("data.img_dir")
    mask_dir    = cfg.get("data.mask_dir")
    batch_size  = cfg.get("data.batch_size", 32)
    num_workers = cfg.get("data.num_workers", 4)

    # ------------------------------
    # Classification path
    # ------------------------------
    if task == "classification":
        cls_aug_cfg = cfg.get("augmentation", {}).get("classification", {})
        mean = cls_aug_cfg.get("mean", [0.485, 0.456, 0.406])
        std  = cls_aug_cfg.get("std",  [0.229, 0.224, 0.225])

        # Training transform
        train_tf = transforms.Compose([
            transforms.Resize(tuple(cls_aug_cfg.get("resize", (256, 256)))),
            transforms.RandomCrop(tuple(cls_aug_cfg.get("crop", (224, 224)))),
            transforms.RandomHorizontalFlip(p=cls_aug_cfg.get("horizontal_flip_prob", 0.5)),
            transforms.RandomVerticalFlip(p=cls_aug_cfg.get("vertical_flip_prob", 0.5)),
            transforms.RandomRotation(degrees=cls_aug_cfg.get("rotation_degrees", 30)),
            transforms.RandomPerspective(
                distortion_scale=cls_aug_cfg.get("elastic_alpha", 0.1),
                p=cls_aug_cfg.get("perspective_prob", 0.5)
            ),
            transforms.ColorJitter(
                brightness=cls_aug_cfg.get("brightness", 1.0),
                contrast=cls_aug_cfg.get("contrast", 1.0),
                saturation=cls_aug_cfg.get("saturation", 1.0),
                hue=cls_aug_cfg.get("hue", 0.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Validation (or "all") transform
        val_tf = transforms.Compose([
            transforms.Resize(tuple(cls_aug_cfg.get("resize", (224, 224)))),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Train loader
        train_ds = MHISTDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            mask_dir=None,
            transform=train_tf,
            partition="train",
            task="classification"
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        # Validation loader
        val_ds = MHISTDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            mask_dir=None,
            transform=val_tf,
            partition="test",
            task="classification"
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader

    # ------------------------------
    # Segmentation path
    # ------------------------------
    else:
        # Raw train dataset (no transforms)
        raw_train_ds = MHISTDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            mask_dir=mask_dir,
            transform=None,
            partition="train",
            task="segmentation"
        )

        # Split into positive (SSA) and negative (HP)
        labels = raw_train_ds.data["label"].astype(int).values
        pos_indices = np.where(labels == 1)[0].tolist()
        neg_indices = np.where(labels == 0)[0].tolist()
        if not pos_indices or not neg_indices:
            raise RuntimeError("Need both positive (SSA) and negative (HP) samples for segmentation.")

        # Raw validation dataset
        raw_val_ds = MHISTDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            mask_dir=mask_dir,
            transform=None,
            partition="test",
            task="segmentation"
        )

        # Albumentations configs
        alb_train_cfg = cfg.get("augmentation", {}).get("segmentation", {}).get("train", {})
        alb_val_cfg   = cfg.get("augmentation", {}).get("segmentation", {}).get("test", {})

        resize = tuple(alb_train_cfg.get("resize", (224, 224)))  # (H, W)

        # Wrappers
        pos_ds = SegTrainWrapper(raw_train_ds, pos_indices, img_dir, mask_dir, alb_train_cfg, resize)
        neg_ds = SegTrainWrapper(raw_train_ds, neg_indices, img_dir, mask_dir, alb_train_cfg, resize)
        val_ds = SegValWrapper(raw_val_ds, list(range(len(raw_val_ds))), img_dir, mask_dir, alb_val_cfg, resize)

        # DataLoaders
        pos_loader = DataLoader(
            pos_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        neg_loader = DataLoader(
            neg_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        return pos_loader, neg_loader, val_loader
