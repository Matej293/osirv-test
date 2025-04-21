import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MHISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, partition="train"):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        if not os.path.exists(img_dir):
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # Filter by partition
        self.data = self.data[self.data["Partition"] == partition]
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

        if self.transform:
            image = self.transform(image)

        label_mask = torch.full((224, 224), label, dtype=torch.long)

        return image, label_mask

def get_mhist_dataloader(csv_file, img_dir, batch_size=16, partition="train", augmentation_config=None, distributed=False, world_size=1, rank=0, return_dataset=False): 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # train transformations
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
        
    # test transformations
    elif partition == "test":
        transform = transforms.Compose([
            transforms.Resize(tuple(augmentation_config.get('resize', (224, 224)))),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    dataset = MHISTDataset(csv_file, img_dir, transform=transform, partition=partition)
    
    # for distributed training, we need to ensure that each process gets a different subset of the data
    if return_dataset:
        return dataset
    
    # Create appropriate dataloader based on whether distributed training is used
    if distributed and partition == "train":
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=(partition=="train"),
            num_workers=4,
            pin_memory=True
        )
    
    return dataloader
