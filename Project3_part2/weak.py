import os
import argparse
import numpy as np
from glob import glob
from PIL import Image
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from datasets import PH2WeakDataset
import matplotlib.pyplot as plt
import json


# =====================================================
# ARGUMENT PARSER
# =====================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Weak Supervision Pipeline for PH2 (click generation + training)"
    )

    parser.add_argument("--generate_clicks", action="store_true",
                        help="Generate weak point-click annotations")

    parser.add_argument("--train", action="store_true",
                        help="Train weak supervision model")

    parser.add_argument("--root", type=str,
                        default="/dtu/datasets1/02516/PH2_Dataset_images",
                        help="Path to PH2 dataset root folder")

    parser.add_argument("--out_dir", type=str,
                        default="weak_clicks",
                        help="Output directory for click .npy files "
                             "(default will be expanded to weak_clicks_posX_negY)")

    parser.add_argument("--pos", type=int, default=5,
                        help="Number of positive clicks per image")

    parser.add_argument("--neg", type=int, default=5,
                        help="Number of negative clicks per image")

    parser.add_argument("--jitter", type=int, default=5,
                        help="Pixel jitter added to simulate human annotation noise")

    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs for the weak model")

    return parser.parse_args()


# =====================================================
# CLICK SAMPLING FUNCTIONS
# =====================================================
def sample_with_jitter(coords, num, jitter, H, W):
    """Vælg 'num' koordinater og tilføj jitter, clip til billedstørrelse."""
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    chosen = coords[np.random.choice(len(coords),
                                     size=min(num, len(coords)),
                                     replace=False)]
    jittered = chosen + np.random.randint(-jitter, jitter + 1, size=chosen.shape)

    # clip
    jittered[:, 0] = np.clip(jittered[:, 0], 0, H - 1)
    jittered[:, 1] = np.clip(jittered[:, 1], 0, W - 1)

    return jittered.astype(np.int32)


def _sample_with_distance(mask_binary, num_clicks, jitter):
    """
    Fælles helper til pos/neg:
    - mask_binary er 0/1 billede (1 = område vi vil sample fra)
    """
    coords = np.argwhere(mask_binary == 1)
    H, W = mask_binary.shape

    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    dist_map = distance_transform_edt(mask_binary)
    flat = dist_map.flatten()
    if flat.sum() > 0:
        probs = flat / flat.sum()
        indices = np.random.choice(len(flat), size=num_clicks,
                                   replace=False, p=probs)
    else:
        # fallback: uniform sampling
        indices = np.random.choice(len(flat), size=num_clicks, replace=False)

    coords_raw = np.column_stack(np.unravel_index(indices, mask_binary.shape))
    return sample_with_jitter(coords_raw, num_clicks, jitter, H, W)


def sample_positive_clicks(mask, num_clicks, jitter):
    # mask==1 er læsion
    return _sample_with_distance((mask == 1).astype(np.uint8), num_clicks, jitter)


def sample_negative_clicks(mask, num_clicks, jitter):
    # mask==0 er baggrund
    return _sample_with_distance((mask == 0).astype(np.uint8), num_clicks, jitter)


# =====================================================
# CLICK GENERATION MAIN
# =====================================================
def generate_clicks(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    cases = sorted(glob(os.path.join(args.root, "IMD*")))

    print(f"\nGenerating weak clicks in '{out_dir}' for {len(cases)} cases...")
    print(f"Positive clicks per image: {args.pos}")
    print(f"Negative clicks per image: {args.neg}")
    print(f"Jitter: {args.jitter}px\n")

    for case_dir in cases:
        name = os.path.basename(case_dir)

        mask_path = os.path.join(case_dir, f"{name}_lesion", f"{name}_lesion.bmp")
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.uint8)

        pos_clicks = sample_positive_clicks(mask, args.pos, args.jitter)
        neg_clicks = sample_negative_clicks(mask, args.neg, args.jitter)

        np.save(os.path.join(out_dir, f"{name}_pos.npy"), pos_clicks)
        np.save(os.path.join(out_dir, f"{name}_neg.npy"), neg_clicks)

        print(f"  {name}: pos={len(pos_clicks)}, neg={len(neg_clicks)}")

    print("\n✓ All weak annotations generated successfully!\n")


# =====================================================
# COLLATE FN
# =====================================================
def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    masks  = torch.stack([item[1] for item in batch])   # bruges til val-metrics
    pos    = [item[2] for item in batch]
    neg    = [item[3] for item in batch]

    return {
        "image": images,
        "mask": masks,
        "pos_clicks": pos,
        "neg_clicks": neg
    }


# =====================================================
# METRICS
# =====================================================
def dice(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    return (2 * inter + eps) / (pred.sum() + target.sum() + eps)


def iou(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


def accuracy(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    return (pred == target).float().mean()


def sensitivity(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(pred, target, eps=1e-6):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tn = ((1 - pred) * (1 - target)).sum()
    fp = (pred * (1 - target)).sum()
    return (tn + eps) / (tn + fp + eps)


# =====================================================
# U-NET
# =====================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 64);  self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128);          self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(128, 256);         self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(256, 512);         self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# =====================================================
# POINT LOSS
# =====================================================
bce = nn.BCEWithLogitsLoss()

def point_loss(logits, pos_clicks, neg_clicks):
    B, _, H, W = logits.shape
    total, valid = 0, 0

    for b in range(B):
        log = logits[b, 0]
        preds, targets = [], []

        # positive clicks
        if len(pos_clicks[b]) > 0:
            ys = torch.tensor(pos_clicks[b][:, 0]).clamp(0, H - 1).long()
            xs = torch.tensor(pos_clicks[b][:, 1]).clamp(0, W - 1).long()
            preds.append(log[ys, xs])
            targets.append(torch.ones(len(ys)))

        # negative clicks
        if len(neg_clicks[b]) > 0:
            ys = torch.tensor(neg_clicks[b][:, 0]).clamp(0, H - 1).long()
            xs = torch.tensor(neg_clicks[b][:, 1]).clamp(0, W - 1).long()
            preds.append(log[ys, xs])
            targets.append(torch.zeros(len(ys)))

        if preds:
            preds_tensor = torch.cat(preds).to(log.device)
            targets_tensor = torch.cat(targets).float().to(log.device)
            total += bce(preds_tensor, targets_tensor)
            valid += 1

    return torch.tensor(0.0, device=logits.device) if valid == 0 else total / valid


# =====================================================
# EVALUATION
# =====================================================
def evaluate(model, loader, device):
    model.eval()
    dice_sum = iou_sum = acc_sum = sens_sum = spec_sum = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            logits = model(img)

            dice_sum += dice(logits, mask).item()
            iou_sum  += iou(logits, mask).item()
            acc_sum  += accuracy(logits, mask).item()
            sens_sum += sensitivity(logits, mask).item()
            spec_sum += specificity(logits, mask).item()
            count += 1

    if count == 0:
        return {"dice": 0, "iou": 0, "acc": 0, "sens": 0, "spec": 0}

    return {
        "dice": dice_sum / count,
        "iou":  iou_sum / count,
        "acc":  acc_sum / count,
        "sens": sens_sum / count,
        "spec": spec_sum / count
    }


# =====================================================
# PLOT RESULTS (weak only, med pos/neg)
# =====================================================
def plot_results(history, final_metrics, args):
    pos, neg = args.pos, args.neg
    print(f"\n{'='*60}")
    print(f"GENERATING PLOTS (pos={pos}, neg={neg})")
    print(f"{'='*60}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(
        f'Weak Supervision U-Net – PH2 (pos={pos}, neg={neg})',
        fontsize=16,
        fontweight='bold'
    )

    epochs = len(history["train_loss"])
    epochs_range = range(1, epochs + 1)

    color = '#e67e22'

    # 1) Training loss
    ax = axes[0, 0]
    ax.plot(epochs_range, history['train_loss'],
            label=f"Weak (pos={pos}, neg={neg})",
            color=color, linewidth=2, marker='o')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(alpha=0.3)

    # 2–6) Val metrics
    metric_keys = ['dice', 'iou', 'acc', 'sens', 'spec']
    titles = ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
    positions = [(0,1), (0,2), (1,0), (1,1), (1,2)]

    for mk, title, pos_ax in zip(metric_keys, titles, positions):
        ax = axes[pos_ax]
        ax.plot(epochs_range, history[f'val_{mk}'],
                label=title, color=color, linewidth=2, marker='o')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(alpha=0.3)

    # 7) Bar chart – "test" (her = val efter træning)
    ax = axes[2, 0]
    labels = ['Dice', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']
    values = [final_metrics[m] for m in metric_keys]
    x = np.arange(len(labels))

    ax.bar(x, values, color=color, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title(f"Final Validation Metrics (pos={pos}, neg={neg})")
    ax.grid(axis='y', alpha=0.3)

    # 8) Summary table
    ax = axes[2, 1]
    ax.axis('off')

    table_data = [
        [f"Weak (pos={pos}, neg={neg})"] +
        [f"{final_metrics[m]:.3f}" for m in metric_keys]
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Model"] + labels,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # 9) tom – kun for layout
    axes[2, 2].axis('off')

    plt.tight_layout()

    plot_name = f"weak_model_results_pos{pos}_neg{neg}.png"
    plt.savefig(plot_name, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✓ Saved plot to '{plot_name}'")


# =====================================================
# TRAIN FUNCTION
# =====================================================
def train_weak(args):
    transform = T.Compose([T.Resize((256, 256)), T.ToTensor()])

    dataset = PH2WeakDataset(args.root, args.out_dir, transform)

    train_len = int(0.8 * len(dataset))
    val_len   = len(dataset) - train_len

    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False,
                              collate_fn=collate_fn)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = UNet().to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = {
        "train_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_acc": [],
        "val_sens": [],
        "val_spec": []
    }

    print(f"\nTraining weak model (pos={args.pos}, neg={args.neg})...\n")

    for epoch in range(args.epochs):
        model.train()
        ep_loss = 0.0

        for batch in train_loader:
            img = batch["image"].to(device)
            pos = batch["pos_clicks"]
            neg = batch["neg_clicks"]

            logits = model(img)
            loss = point_loss(logits, pos, neg)

            optim.zero_grad()
            loss.backward()
            optim.step()

            ep_loss += loss.item()

        ep_loss /= len(train_loader)
        history["train_loss"].append(ep_loss)

        metrics = evaluate(model, val_loader, device)
        for k in metrics:
            history[f"val_{k}"].append(metrics[k])

        print(f"Epoch {epoch+1:02d}/{args.epochs} | "
              f"Loss={ep_loss:.4f} | Dice={metrics['dice']:.4f}")

    model_name = f"weak_unet_pos{args.pos}_neg{args.neg}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"\n✓ Model saved as {model_name}\n")

    with open(f'results_arrays/results_p{args.pos}_n{args.neg}.json', 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

    # sidste metrics = "final validation metrics"
    final_metrics = evaluate(model, val_loader, device)
    return history, final_metrics


# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    args = parse_args()

    # Default: hvis man IKKE angiver --generate_clicks / --train,
    # så gør vi begge dele for de givne pos/neg.
    if not args.generate_clicks and not args.train:
        args.generate_clicks = True
        args.train = True

    # Lav automatisk mappenavn pr. konfiguration hvis brugeren
    # ikke specifikt har valgt noget andet
    if args.out_dir == "weak_clicks":
        args.out_dir = f"weak_clicks_pos{args.pos}_neg{args.neg}"

    if args.generate_clicks:
        generate_clicks(args)

    if args.train:
        history, final_metrics = train_weak(args)
        plot_results(history, final_metrics, args)
        print("\nFinal weak validation metrics:")
        print(final_metrics)
