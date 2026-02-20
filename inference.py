import argparse
import os
import glob
import numpy as np
from PIL import Image
import cv2
import torch
from torch.cuda import amp
from model.mobilevit_aspp import MobileViT_ASPP
from datasets.dataset_original import get_val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="MobileViT-ASPP Inference")

    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)

    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--model_name', type=str, default="mobilevit_s")
    parser.add_argument('--threshold', type=float, default=0.5)

    return parser.parse_args()

# Post-processing function for predicted masks
def modify_mask(pred_mask):
    mask_uint8 = (pred_mask * 255).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

    blurred = cv2.GaussianBlur(closed, (11, 11), 0)

    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            cv2.drawContours(thresholded, [cnt], 0, 0, -1)

    smoothed = cv2.medianBlur(thresholded, 5)

    return (smoothed > 0).astype(np.uint8)


# Inference Function
@torch.no_grad()
def predict(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MobileViT_ASPP(
        model_name=args.model_name,
        num_classes=1
    ).to(device)

    state = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state['model'])
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    transform = get_val_transform(args.image_size)

    image_paths = sorted([
        p for p in glob.glob(os.path.join(args.input_dir, "*"))
        if p.lower().endswith((".jpg", ".png"))
    ])

    print(f"Found {len(image_paths)} images")

    with amp.autocast(enabled=(device.type == "cuda")):
        for img_path in image_paths:

            img = np.array(Image.open(img_path).convert("RGB"))
            aug = transform(image=img)
            tensor_img = aug["image"].unsqueeze(0).to(device)

            logits = model(tensor_img)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            processed_mask = modify_mask(probs)

            save_path = os.path.join(args.output_dir, os.path.basename(img_path))
            Image.fromarray((processed_mask * 255).astype(np.uint8)).save(save_path)

            print(f"Saved: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    predict(args)