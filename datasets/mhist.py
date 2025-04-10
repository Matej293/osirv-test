import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class MHISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, partition="train"):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.partition = partition

        # partition the dataset into train/test
        self.data = self.data[self.data["Partition"] == partition]

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


def get_mhist_dataloader(csv_file, img_dir, batch_size=16, partition="train"): 
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if partition == "train":
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.3, hue=0.05),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    dataset = MHISTDataset(csv_file, img_dir, transform=transform, partition=partition)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(partition=="train"))
    return dataloader
