import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class MHISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, transform=None, partition="train", task="classification"):
        """
        MHIST dataset that supports both classification and segmentation.
        
        Args:
            csv_file (str): Path to the csv file with annotations.
            img_dir (str): Directory with all the images.
            mask_dir (str, optional): Directory with mask images for segmentation task. Required if task="segmentation".
            transform (callable, optional): Optional transform to be applied on images.
            partition (str): Data partition - "train", "test", or "all" (classification only).
            task (str): "classification" or "segmentation".
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if task == "segmentation" and (mask_dir is None or not os.path.exists(mask_dir)):
            raise FileNotFoundError(f"Mask directory required for segmentation task")
            
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.task = task
        
        # Filter by partition
        if partition in ["train", "test"]:
            self.data = self.data[self.data["Partition"] == partition]
        elif partition == "all" and task == "classification":
            pass  # Use all data
        else:
            raise ValueError(f"Invalid partition: {partition}" + 
                            (f" for task: {task}" if partition == "all" else ""))
            
        if len(self.data) == 0:
            raise ValueError(f"No samples found for partition: {partition}")
            
        # Map class labels
        self.label_map = {"SSA": 1, "HP": 0}
        self.data["Majority Vote Label"] = self.data["Majority Vote Label"].map(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Name"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = self.data.iloc[idx]["Majority Vote Label"]

        # Apply transform to image
        if self.transform:
            image = self.transform(image)
            
        # Return different outputs based on task
        if self.task == "classification":
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        else:  # segmentation
            # Load mask
            mask_name = img_name.replace('.png', '_mask.png')
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = Image.open(mask_path).convert("L")  # greyscale
            
            # Process mask
            mask_np = np.array(mask)
            mask_binary = (mask_np > 50).astype(np.float32)
            mask = torch.tensor(mask_binary, dtype=torch.float)
            
            return image, mask


def get_mhist_dataloader(
    csv_file, 
    img_dir, 
    mask_dir=None, 
    batch_size=32, 
    partition="train", 
    augmentation_config=None,
    task="classification"
): 
    """
    Create a dataloader for MHIST dataset.
    
    Args:
        csv_file (str): Path to CSV file with annotations
        img_dir (str): Directory with images
        mask_dir (str, optional): Directory with masks (required for segmentation)
        batch_size (int): Batch size
        partition (str): "train", "test", or "all" (classification only)
        augmentation_config (dict): Configuration for data augmentation
        task (str): "classification" or "segmentation"
    """
    if augmentation_config is None:
        augmentation_config = {}
        
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if task == "classification":
        # Classification transformations
        if partition == "train":
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize', (256, 256)))),
                transforms.RandomCrop(tuple(augmentation_config.get('crop', (224, 224)))),
                transforms.RandomHorizontalFlip(p=augmentation_config.get('horizontal_flip_prob', 0.5)),
                transforms.RandomVerticalFlip(p=augmentation_config.get('vertical_flip_prob', 0.5)),
                transforms.RandomRotation(degrees=augmentation_config.get('rotation_degrees', 30)),
                transforms.ColorJitter(
                    brightness=augmentation_config.get('brightness', 1.0),
                    contrast=augmentation_config.get('contrast', 1.0),
                    saturation=augmentation_config.get('saturation', 1.0),
                    hue=augmentation_config.get('hue', 0.0)
                ),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=tuple(augmentation_config.get('translate', (0.1, 0.1)))
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:  # test or all
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize', (224, 224)))),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    else:  # segmentation
        # Segmentation transformations
        if partition == "train":
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize', (224, 224)))),
                transforms.ColorJitter(
                    brightness=augmentation_config.get('brightness', 1.0),
                    contrast=augmentation_config.get('contrast', 1.0),
                    saturation=augmentation_config.get('saturation', 1.0),
                    hue=augmentation_config.get('hue', 0.0)
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        else:  # test
            transform = transforms.Compose([
                transforms.Resize(tuple(augmentation_config.get('resize', (224, 224)))),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
    
    dataset = MHISTDataset(
        csv_file, 
        img_dir, 
        mask_dir=mask_dir, 
        transform=transform, 
        partition=partition,
        task=task
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(partition=="train"),
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader