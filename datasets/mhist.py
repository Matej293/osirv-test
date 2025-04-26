import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

class MHISTDataset(Dataset):
    def __init__(self,
                 csv_file: str,
                 img_dir: str,
                 mask_dir: str = None,
                 transform = None,
                 partition: str = "train",
                 task: str = "classification"):
        """
        MHIST dataset supporting classification or segmentation.
        For segmentation, returns (img_tensor, gt_mask, tissue_mask).
        """
        # basic existence checks
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if task=="segmentation" and (mask_dir is None or not os.path.exists(mask_dir)):
            raise FileNotFoundError(f"Mask directory required for segmentation task")
        
        self.data      = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.task      = task

        # partition filter
        if partition in ("train","test"):
            self.data = self.data[self.data["Partition"]==partition]
        elif partition=="all":
            pass
        else:
            raise ValueError(f"Invalid partition={partition}")
        if self.data.empty:
            raise ValueError(f"No samples in partition={partition}")

        # map labels
        self.data["label"] = self.data["Majority Vote Label"].map({"HP":0,"SSA":1})

        # for segmentation, detect the resize in transform
        self.mask_size = None
        if task=="segmentation":
            if isinstance(transform, transforms.Compose):
                for t in transform.transforms:
                    if isinstance(t, transforms.Resize):
                        sz = t.size
                        if isinstance(sz,int):
                            sz = (sz,sz)
                        self.mask_size = tuple(sz)
                        break
            if self.mask_size is None:
                raise ValueError("Segmentation transform must include a Resize")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row   = self.data.iloc[idx]
        fname = row["Image Name"]
        img_pil = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")

        # --- compute tissue mask from raw RGB ---
        # 1) convert to numpy HxWx3 uint8
        img_np = np.array(img_pil)
        # 2) grayscale and normalize
        gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) / 255.0
        # 3) threshold white background
        tissue_np = (gray < 0.95).astype(np.uint8)    # HxW, 1=tissue,0=bg

        # --- apply torchvision transform to the PIL image only ---
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = transforms.ToTensor()(img_pil)

        if self.task=="classification":
            label = torch.tensor(row["label"], dtype=torch.long)
            return img_tensor, label

        # --- segmentation branch: load GT mask & resize to same mask_size ---
        mask_name = fname.replace(".png","_mask.png")
        mask_pil  = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")
        mask_pil  = mask_pil.resize(self.mask_size, resample=Image.NEAREST)
        mask_np   = np.array(mask_pil)
        gt_mask   = (mask_np > 50).astype(np.float32)  # HxW
        gt_mask   = torch.from_numpy(gt_mask)[None]    # 1xHxW

        # --- resize tissue_np to mask_size via PIL nearest ---
        tissue_img = Image.fromarray((tissue_np*255).astype(np.uint8))
        tissue_img = tissue_img.resize(self.mask_size, resample=Image.NEAREST)
        tissue_mask = (np.array(tissue_img)>128).astype(np.float32)
        tissue_mask = torch.from_numpy(tissue_mask)[None]  # 1xHxW

        return img_tensor, gt_mask, tissue_mask


def get_mhist_dataloader(
    csv_file, img_dir, mask_dir=None,
    batch_size=32, partition="train",
    augmentation_config=None, task="classification"
):
    """
    Returns a DataLoader for MHIST.
    For segmentation: each item is (img_tensor, gt_mask, tissue_mask).
    """
    if augmentation_config is None:
        augmentation_config = {}

    mean = [0.485,0.456,0.406]
    std  = [0.229,0.224,0.225]

    if task=="classification":
        if partition=="train":
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize',(256,256)))),
                transforms.RandomCrop(tuple(augmentation_config.get('crop',(224,224)))),
                transforms.RandomHorizontalFlip(p=augmentation_config.get('horizontal_flip_prob',0.5)),
                transforms.RandomVerticalFlip(p=augmentation_config.get('vertical_flip_prob',0.5)),
                transforms.RandomRotation(degrees=augmentation_config.get('rotation_degrees',30)),
                transforms.RandomPerspective(
                    distortion_scale=augmentation_config.get('elastic_alpha',0.1),
                    p=0.5
                ),
                transforms.ColorJitter(
                    brightness=augmentation_config.get('brightness',1.0),
                    contrast=augmentation_config.get('contrast',1.0),
                    saturation=augmentation_config.get('saturation',1.0),
                    hue=augmentation_config.get('stain_jitter',0.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize',(224,224)))),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
    else:  # segmentation
        if partition=="train":
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize',(224,224)))),
                transforms.ColorJitter(
                    brightness=augmentation_config.get('brightness',1.0),
                    contrast=augmentation_config.get('contrast',1.0),
                    saturation=augmentation_config.get('saturation',1.0),
                    hue=augmentation_config.get('hue',0.0)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize',(224,224)))),
                transforms.ToTensor(),
                transforms.Normalize(mean,std)
            ])

    dataset = MHISTDataset(
        csv_file  = csv_file,
        img_dir   = img_dir,
        mask_dir  = mask_dir,
        transform = transform,
        partition = partition,
        task      = task
    )

    loader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = (partition=="train"),
        num_workers= 4,
        pin_memory = True
    )
    return loader
