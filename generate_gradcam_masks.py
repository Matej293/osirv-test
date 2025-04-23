import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.mhist import MHISTDataset
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# === Paths & settings ===
model_path = './models/model_qdpxjvvw.pth'
image_dir = './data/images'
csv_path = './data/mhist_annotations.csv'
output_dir = './data/pseudo_masks'
os.makedirs(output_dir, exist_ok=True)
batch_size = 1  # Grad-CAM is usually per-image
threshold = 0.3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
from network import modeling  # Make sure this matches your import path

model = modeling.deeplabv3plus_resnet101(
    num_classes=1,
    output_stride=16,
    pretrained_backbone=True,
)

checkpoint = torch.load(model_path, map_location=device)
# if 'model_state' in checkpoint:
#     state_dict = checkpoint['model_state']
#     state_dict.pop('classifier.classifier.3.weight', None)
#     state_dict.pop('classifier.classifier.3.bias', None)
#     model.load_state_dict(state_dict, strict=False)
#     print(f"Loaded model from {model_path}")
# else:
#     print(f"Warning: Invalid checkpoint format in {model_path}")

model.classifier = modeling.DeepLabHeadV3Plus(
    in_channels=2048,
    low_level_channels=256,
    num_classes=1,
    aspp_dilate=[12, 24, 36]
)

model.to(device).eval()

# === Dataset setup ===
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset = MHISTDataset(csv_file=csv_path, img_dir=image_dir, transform=transform, partition="all")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# === Grad-CAM setup ===
target_layers = [model.backbone.layer4[-1]]  # Use deepest conv layer for Grad-CAM
cam = GradCAM(model=model, target_layers=target_layers)

# === Mask saving function ===
def save_mask(mask, output_path):
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_path, mask)

class MeanSegmentationTarget:
    def __call__(self, model_output):
        return model_output.mean()
    
# === Generate masks ===
print("Generating Grad-CAM masks...")
for i, (image, label) in enumerate(tqdm(dataloader)):
    image = image.to(device)

    #outputs = model(image)  # [B, 1, H, W]
    grayscale_cam = cam(input_tensor=image, targets=[MeanSegmentationTarget()])[0]

    # Optional: threshold to create binary mask
    # binary_mask = (grayscale_cam > threshold).astype(np.uint8)

    img_name = dataset.data.iloc[i]["Image Name"]
    base_name = os.path.splitext(img_name)[0]
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    save_mask(grayscale_cam, mask_path)

print(f"Saved all Grad-CAM masks to: {output_dir}")
