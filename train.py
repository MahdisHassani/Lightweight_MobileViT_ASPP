import argparse
import time
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp
from model.mobilevit_aspp import MobileViT_ASPP
from datasets.dataset_original import SegmentationDataset, get_train_transform, get_val_transform
from datasets.dataset_compressed import SegDatasetCompressed
from utils import set_seed, BCEDiceLoss, compute_batch_metrics, save_checkpoint, load_last_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="MobileViT-ASPP Training and Evaluation")

    parser.add_argument('--dataset_type', type=str, choices=['original', 'compressed'], required=True)

    parser.add_argument('--train_image_dir', type=str, required=True)
    parser.add_argument('--train_mask_dir', type=str, required=True)
    parser.add_argument('--val_image_dir', type=str, required=True)
    parser.add_argument('--val_mask_dir', type=str, required=True)
    parser.add_argument('--test_image_dir', type=str, required=True)
    parser.add_argument('--test_mask_dir', type=str, required=True)

    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--threshold', type=float, default=0.5)

    return parser.parse_args()


# Training Loop

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch [{epoch}/{total_epochs}] Train", leave=True)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
        })

    return running_loss / len(loader.dataset)


# Validation / Test Loop

@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold, mode="Val"):
    model.eval()
    running_loss = 0.0
    metrics_sum = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'iou': 0.0}
    count_batches = 0

    pbar = tqdm(loader, desc=f"{mode} Evaluation", leave=True)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with amp.autocast(enabled=(device.type == "cuda")):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)

        p, r, f1, iou = compute_batch_metrics(logits, masks, threshold=threshold)
        metrics_sum['precision'] += p
        metrics_sum['recall'] += r
        metrics_sum['f1'] += f1
        metrics_sum['iou'] += iou
        count_batches += 1

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    val_loss = running_loss / len(loader.dataset)
    for k in metrics_sum:
        metrics_sum[k] /= max(count_batches, 1)

    return val_loss, metrics_sum


# Main Function

def main():
    args = parse_args()
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset Selection
    if args.dataset_type == "original":
        train_ds = SegmentationDataset(
            args.train_image_dir,
            args.train_mask_dir,
            transform=get_train_transform(args.image_size)
        )
        val_ds = SegmentationDataset(
            args.val_image_dir,
            args.val_mask_dir,
            transform=get_val_transform(args.image_size)
        )
        test_ds = SegmentationDataset(
            args.test_image_dir,
            args.test_mask_dir,
            transform=get_val_transform(args.image_size)
        )
    else:
        train_ds = SegDatasetCompressed(
            args.train_image_dir,
            args.train_mask_dir,
            transform=get_train_transform(args.image_size),
            mode="train"
        )
        val_ds = SegDatasetCompressed(
            args.val_image_dir,
            args.val_mask_dir,
            transform=get_val_transform(args.image_size),
            mode="val"
        )
        test_ds = SegDatasetCompressed(
            args.test_image_dir,
            args.test_mask_dir,
            transform=get_val_transform(args.image_size),
            mode="test"
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)


    # Model, Loss, Optimizer

    model = MobileViT_ASPP(model_name="mobilevit_s", num_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    scaler = amp.GradScaler()

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion,
                                     optimizer, scaler, device,
                                     epoch, args.epochs)

        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, args.threshold, mode="Val")

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Precision:  {val_metrics['precision']:.4f}")
        print(f"Recall:     {val_metrics['recall']:.4f}")
        print(f"F1-Score:   {val_metrics['f1']:.4f}")
        print(f"mIoU:       {val_metrics['iou']:.4f}")
        print(f"Time:       {(time.time() - t0):.1f}s")

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_val': best_val
        }

        save_checkpoint(state, args.output_dir, epoch, is_best)


    # Test Evaluation

    print("\n=== Running Test Evaluation on Best Model ===")

    checkpoint_path = os.path.join(args.output_dir, "best.pth")
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state['model'])
        test_loss, test_metrics = evaluate(model, test_loader, criterion, device, args.threshold, mode="Test")
        print(f"Test Loss:  {test_loss:.4f}")
        print(f"Precision:  {test_metrics['precision']:.4f}")
        print(f"Recall:     {test_metrics['recall']:.4f}")
        print(f"F1-Score:   {test_metrics['f1']:.4f}")
        print(f"mIoU:       {test_metrics['iou']:.4f}")
    else:
        print("No best model checkpoint found for test evaluation.")


if __name__ == "__main__":
    main()