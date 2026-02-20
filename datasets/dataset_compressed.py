import numpy as np
from PIL import Image
import glob
import os
import io
import random
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training data augmentation pipeline

def get_train_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])

# Validation transformation

def get_val_transform(size):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()])


# Dataset with JPEG Compression

class SegDatasetCompressed(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mode="train"):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*")))
        self.transform = transform
        self.mode = mode 
        # JPEG quality levels
        self.quality_levels = [20, 30, 40]
        
        if self.mode in ["val", "test"]:
            self.fixed_qualities = [random.choice(self.quality_levels) for _ in range(len(self.img_paths))]

    def jpeg_compress(self, img_np, quality):
        pil_img = Image.fromarray(img_np)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        return np.array(compressed_img)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]).convert("L"))
        mask = (mask > 0).astype(np.float32)

        if self.mode == "train":
            q = random.choice(self.quality_levels)
        else:
            q = self.fixed_qualities[idx]

        img = self.jpeg_compress(img, quality=q)

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img.float(), mask.float()