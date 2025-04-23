import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from datasets.mhist import get_mhist_dataloader
from pytorch_grad_cam import GradCAM

def main():
    # === Paths & settings ===
    model_path = './models/model_qdpxjvvw.pth'
    image_dir = './data/images'
    csv_path = './data/mhist_annotations.csv'
    output_dir = './data/pseudo_masks'
    os.makedirs(output_dir, exist_ok=True)
    batch_size = 1  # Grad-CAM is usually per-image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    from network import modeling

    model = modeling.deeplabv3plus_resnet101(
        num_classes=1,
        output_stride=16,
        pretrained_backbone=True,
    )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        state_dict.pop('classifier.classifier.3.weight', None)
        state_dict.pop('classifier.classifier.3.bias', None)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Invalid checkpoint format in {model_path}")

    model.classifier = modeling.DeepLabHeadV3Plus(
        in_channels=2048,
        low_level_channels=256,
        num_classes=1,
        aspp_dilate=[6, 12, 18]
    )

    model.to(device).eval()

    # === Dataset & Dataloader setup ===
    dataloader = get_mhist_dataloader(
        csv_file=csv_path,
        img_dir=image_dir,
        batch_size=batch_size,
        partition="all",
        task="classification"
    )

    # Get direct access to the dataset for indexing
    dataset = dataloader.dataset

    # === Grad-CAM setup ===
    target_layers = [model.backbone.layer4[-1]]  # using deepest conv layer for Grad-CAM
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
    for i, (image, _) in enumerate(tqdm(dataloader)):
        image = image.to(device)

        grayscale_cam = cam(input_tensor=image, targets=[MeanSegmentationTarget()])[0]

        img_name = dataset.data.iloc[i]["Image Name"]
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
        save_mask(grayscale_cam, mask_path)

    print(f"Saved all Grad-CAM masks to: {output_dir}")

if __name__ == "__main__":
    main()
