import os
import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Combination of Binary Cross Entropy and Dice Loss
       
class BCEDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        dice_loss = 1 - (2. * intersection + self.smooth) / (torch.sum(probs) + torch.sum(targets) + self.smooth)
        return bce_loss + dice_loss

# Compute Evaluation Metrics

@torch.no_grad()
def compute_batch_metrics(seg_logits, targets, threshold=0.5):
    probs = torch.sigmoid(seg_logits)
    preds = (probs > threshold).float()
    targets = targets.float()

    inter_pos = (preds * targets).sum()
    union_pos = preds.sum() + targets.sum() - inter_pos
    iou_pos = inter_pos / (union_pos + 1e-7)

    preds_neg = 1 - preds
    targets_neg = 1 - targets
    inter_neg = (preds_neg * targets_neg).sum()
    union_neg = preds_neg.sum() + targets_neg.sum() - inter_neg
    iou_neg = inter_neg / (union_neg + 1e-7)

    miou = (iou_pos + iou_neg) / 2

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    precision = tp / (tp + fp + 1e-7)
    recall    = tp / (tp + fn + 1e-7)
    f1        = 2 * precision * recall / (precision + recall + 1e-7)

    return precision.item(), recall.item(), f1.item(), miou.item()

# Checkpoint Saving

def save_checkpoint(state, output_dir, epoch, is_best=False):
    os.makedirs(output_dir, exist_ok=True)

    epoch_path = os.path.join(output_dir, f"epoch_{epoch:03d}.pth")
    torch.save(state, epoch_path)

    last_path = os.path.join(output_dir, "last.pth")
    torch.save(state, last_path)

    if is_best:
        best_path = os.path.join(output_dir, "best.pth")
        torch.save(state, best_path)

# Resume Training from Last Checkpoint

def load_last_checkpoint(model, output_dir, optimizer=None, scheduler=None, scaler=None):
    last_path = os.path.join(output_dir, "last.pth")

    if not os.path.exists(last_path):
        return 0, float('inf')

    ckpt = torch.load(last_path, map_location='cpu')
    model.load_state_dict(ckpt['model'])

    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])

    start_epoch = ckpt.get('epoch', 0)
    best_val = ckpt.get('best_val', float('inf'))

    print(f"Loaded last checkpoint: epoch {start_epoch}, best_val {best_val:.6f}")

    return start_epoch, best_val