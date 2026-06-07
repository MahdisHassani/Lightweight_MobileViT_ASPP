import argparse
import torch
from torch.utils.data import DataLoader

from model.mobilevit_aspp import MobileViT_ASPP
from datasets.dataset_original import SegmentationDataset, get_val_transform
from datasets.dataset_compressed import SegDatasetCompressed
from utils import BCEDiceLoss, compute_batch_metrics


def parse_args():
    parser = argparse.ArgumentParser("Evaluate MobileViT-ASPP")

    parser.add_argument("--dataset_type", type=str, choices=["original", "compressed"], required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold):

    model.eval()
    running_loss = 0.0

    metrics_sum = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "iou": 0.0}

    num_batches = 0

    for images, masks in loader:

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):

            logits = model(images)

            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)

        p, r, f1, iou = compute_batch_metrics(
            logits,
            masks,
            threshold=threshold)

        metrics_sum["precision"] += p
        metrics_sum["recall"] += r
        metrics_sum["f1"] += f1
        metrics_sum["iou"] += iou

        num_batches += 1

    loss = running_loss / len(loader.dataset)

    for k in metrics_sum:
        metrics_sum[k] /= max(num_batches, 1)

    return loss, metrics_sum


def main():

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset_type == "original":

        dataset = SegmentationDataset(
            args.image_dir,
            args.mask_dir,
            transform=get_val_transform(args.image_size))

    else:

        dataset = SegDatasetCompressed(
            args.image_dir,
            args.mask_dir,
            transform=get_val_transform(args.image_size),
            mode="test")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True)

    model = MobileViT_ASPP(
        model_name="mobilevit_s",
        num_classes=1).to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device)

    model.load_state_dict(checkpoint["model"])

    criterion = BCEDiceLoss()

    loss, metrics = evaluate(
        model,
        loader,
        criterion,
        device,
        args.threshold)

    print("\n===== TEST RESULTS =====")
    print(f"Loss      : {loss:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1']:.4f}")
    print(f"mIoU      : {metrics['iou']:.4f}")


if __name__ == "__main__":
    main()