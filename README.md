# Lightweight_MobileViT_ASPP
Official implementation of the paper: 
"Lightweight Vision Transformer with Atrous Spatial Pyramid Pooling for Accurate Image Inpainting Detection and Localization" 
Submitted to The Visual Computer.

## Python Version

This project was developed using:

- Python 3.10

---

## Installation

Clone this repository:

```bash
git clone https://github.com/MahdisHassani/Lightweight_MobileViT_ASPP.git
cd Lightweight_MobileViT_ASPP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset split used in this project:

- 20,000 images → Training
- 4,000 images → Validation
- 4,000 images → Testing

Download dataset from:

https://drive.google.com/drive/folders/1twZWQXmyP4bz4GqJs1JKWDTV6Rclagke?usp=sharing

---

## Training

Train on original images:

```bash
python train.py \
--dataset_type original \
--train_image_dir dataset/train/images \
--train_mask_dir dataset/train/masks \
--val_image_dir dataset/val/images \
--val_mask_dir dataset/val/masks \
--test_image_dir dataset/test/images \
--test_mask_dir dataset/test/masks
```

Train on JPEG-compressed images:

```bash
python train.py \
--dataset_type compressed \
--train_image_dir dataset/train/images \
--train_mask_dir dataset/train/masks \
--val_image_dir dataset/val/images \
--val_mask_dir dataset/val/masks \
--test_image_dir dataset/test/images \
--test_mask_dir dataset/test/masks
```

---

## Inference

Generate masks using a trained model:

```bash
python inference.py \
--input_dir path/to/images \
--output_dir path/to/save/masks \
--checkpoint_path checkpoints/best.pth
```

---

## Outputs

During training:

- Best model saved as: checkpoints/best.pth
- Validation metrics shown after each epoch
- Test evaluation runs automatically after training

Metrics reported:

- Precision
- Recall
- F1-Score
- mIoU

---

### This code is directly related to the manuscript currently submitted to The Visual Computer.
### If you use this code, please cite the corresponding paper.
