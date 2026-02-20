import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training data augmentation pipeline

def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])

# Validation transformation

def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])


# Custom Dataset class

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, binarize=True):
        self.image_dir = Path(image_dir)
        self.mask_dir  = Path(mask_dir)
        self.transform = transform
        self.images = sorted([p for p in self.image_dir.glob("*") if p.is_file()])
        self.masks  = sorted([p for p in self.mask_dir.glob("*") if p.is_file()])
        assert len(self.images) == len(self.masks), "Number of images and masks must match"
        self.binarize = binarize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path  = self.images[idx]
        mask_path = self.masks[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask  = np.array(Image.open(mask_path).convert("L"))

        if self.binarize:
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = (mask / 255.0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask  = aug["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.float()
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image.float(), mask