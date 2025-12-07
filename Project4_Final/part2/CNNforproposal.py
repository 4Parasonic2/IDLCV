#!/usr/bin/env python3
"""
train_and_eval_proposals.py

- Builds dataloaders from labeled proposal JSON and splits.json
- Fine-tunes a ResNet-18 classifier for proposals (num_classes = inferred from labels)
- Evaluates classification accuracy and per-class precision/recall on validation set
- Safe checkpointing and informative printing
- GPU-optimized with mixed precision training
- Plots training curves

Usage:
    python train_and_eval_proposals.py

Edit the CONFIG block below to tune hyperparameters and paths.
"""
import os
import json
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# -----------------------
# CONFIG - change paths/hyperparams here
# -----------------------
DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
END_PATH = "/zhome/99/4/213789/IDLCV/finalprojecy"
SPLITS_PATH = os.path.join(END_PATH, "splits.json")
LABELED_PROPOSALS_PATH = os.path.join(END_PATH, "labeled_proposals_train.json")
LABELED_PROPOSALS_VAL = os.path.join(END_PATH, "labeled_proposals_val.json")  # optional, may not exist

CKPT_DIR = os.path.join(END_PATH, "proposal_classifier_ckpts")
os.makedirs(CKPT_DIR, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 64  # Increased for GPU
NUM_EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
PIN_MEMORY = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sampling strategy for training: "per_image" (recommended), "global_weighted", or "all"
TRAIN_SAMPLING = "per_image"
K_PER_IMAGE = 8
POS_FRAC = 0.5

# Whether to compute class-weighted CrossEntropyLoss (recommended when imbalance remains)
USE_CLASS_WEIGHTED_LOSS = True

# Validation options
# - val_mode "all": evaluate on all proposals (true metric but slow)
# - val_mode "sampled": sample at most val_k proposals per image (faster)
VAL_MODE = "all"   # "all" or "sampled"
VAL_K_PER_IMAGE = 50   # used if VAL_MODE == "sampled"

# For fast debug runs set DEBUG_LIMIT_IMAGES (None for full dataset)
DEBUG_LIMIT_TRAIN_IMAGES = None
DEBUG_LIMIT_VAL_IMAGES = None

# Logging & checkpointing
PRINT_EVERY_BATCHES = 20
SAVE_BEST_ONLY = True  # if True only keeps best model as best_model.pth

# Validation frequency (validate every N epochs)
VAL_EVERY_N_EPOCHS = 1  # Set to 2 or 3 to speed up training

# Mixed precision training
USE_MIXED_PRECISION = True

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Check GPU availability
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU found! Training will be very slow.")

# -----------------------
# Helpers
# -----------------------
def _box_to_int_coords(box):
    return [int(round(float(x))) for x in box]

# -----------------------
# Dataset
# -----------------------
class ProposalDataset(Dataset):
    """
    Dataset returning (patch_tensor, label, meta) where meta optionally contains filename/box if needed.
    Modes:
      - "all": include all proposals
      - "per_image_sample": sample up to k_per_image proposals per image with pos_frac positives
      - "sampled_for_val": sample up to k_per_image proposals per image (for fast val)
    """
    def __init__(self,
                 images_dir: str,
                 image_list: List[str],
                 labeled_proposals: Dict[str, List[dict]],
                 mode: str = "all",
                 transform=None,
                 k_per_image: int = 8,
                 pos_frac: float = 0.5):
        self.images_dir = images_dir
        self.image_list = list(image_list)
        self.labeled_proposals = labeled_proposals
        self.mode = mode
        self.transform = transform
        self.k_per_image = k_per_image
        self.pos_frac = pos_frac
        self.examples = []  # list of (fname, box, label)

        if self.mode == "all":
            for fname in self.image_list:
                props = labeled_proposals.get(fname, [])
                for p in props:
                    self.examples.append((fname, p["box"], int(p["label"])))
        elif self.mode == "per_image_sample" or self.mode == "sampled_for_val":
            rnd = random.Random(SEED)
            for fname in self.image_list:
                props = labeled_proposals.get(fname, [])
                if not props:
                    continue
                positives = [p for p in props if int(p["label"]) != 0]
                negatives = [p for p in props if int(p["label"]) == 0]

                # for training we aim for pos_frac positives
                if self.mode == "per_image_sample":
                    num_pos = min(len(positives), int(round(self.k_per_image * self.pos_frac)))
                    num_neg = min(len(negatives), self.k_per_image - num_pos)
                else:
                    # for sampled val just cap total proposals
                    num_pos = min(len(positives), self.k_per_image)
                    num_neg = min(len(negatives), max(0, self.k_per_image - num_pos))

                sampled_pos = rnd.sample(positives, num_pos) if num_pos > 0 and len(positives) > 0 else []
                sampled_neg = rnd.sample(negatives, num_neg) if num_neg > 0 and len(negatives) > 0 else []
                pool = sampled_pos + sampled_neg
                for p in pool:
                    self.examples.append((fname, p["box"], int(p["label"])))
        else:
            raise ValueError("Unknown mode for ProposalDataset")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        fname, box, label = self.examples[idx]
        img_path = os.path.join(self.images_dir, fname)
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            xmin, ymin, xmax, ymax = _box_to_int_coords(box)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img.width, xmax)
            ymax = min(img.height, ymax)

            if xmax <= xmin or ymax <= ymin:
                # fallback center small crop
                cx, cy = img.width // 2, img.height // 2
                half = max(1, min(img.width, img.height) // 8)
                crop = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                crop = img.crop((xmin, ymin, xmax, ymax))

            if self.transform:
                patch = self.transform(crop)
            else:
                t = T.Compose([
                    T.Resize((IMG_SIZE, IMG_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                patch = t(crop)
        return patch, label

# -----------------------
# Build dataloaders
# -----------------------
def build_dataloaders(train_sampling: str = TRAIN_SAMPLING,
                      k_per_image: int = K_PER_IMAGE,
                      pos_frac: float = POS_FRAC,
                      val_mode: str = VAL_MODE,
                      val_k_per_image: int = VAL_K_PER_IMAGE,
                      batch_size: int = BATCH_SIZE,
                      num_workers: int = NUM_WORKERS,
                      pin_memory: bool = PIN_MEMORY,
                      debug_limit_train: Optional[int] = DEBUG_LIMIT_TRAIN_IMAGES,
                      debug_limit_val: Optional[int] = DEBUG_LIMIT_VAL_IMAGES
                      ):
    # load splits
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    train_files = splits.get("train", [])
    val_files = splits.get("val", []) or splits.get("validation", []) or []

    # debug limits
    if debug_limit_train:
        train_files = train_files[:debug_limit_train]
    if debug_limit_val:
        val_files = val_files[:debug_limit_val]

    # load labeled proposals
    with open(LABELED_PROPOSALS_PATH, "r") as f:
        labeled_train = json.load(f)

    labeled_val = {}
    if os.path.exists(LABELED_PROPOSALS_VAL):
        with open(LABELED_PROPOSALS_VAL, "r") as f:
            labeled_val = json.load(f)
    else:
        for fname in val_files:
            if fname in labeled_train:
                labeled_val[fname] = labeled_train[fname]
        if len(labeled_val) == 0:
            # fallback: sample some labeled images for val
            available = list(labeled_train.keys())
            if len(available) > 0:
                rnd = random.Random(SEED+1)
                sample_n = min(len(available), max(1, int(0.1 * len(available))))
                sampled = rnd.sample(available, sample_n)
                for fname in sampled:
                    labeled_val[fname] = labeled_train[fname]
                val_files = list(set(val_files + sampled))
                train_files = [f for f in train_files if f not in sampled]

    # infer num_classes
    all_labels = set()
    for v in labeled_train.values():
        for p in v:
            all_labels.add(int(p["label"]))
    all_labels.add(0)
    num_classes = max(all_labels) + 1

    # transforms
    train_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.05)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # train dataset
    if train_sampling == "per_image":
        train_dataset = ProposalDataset(IMAGES_DIR, train_files, labeled_train,
                                        mode="per_image_sample", transform=train_transform,
                                        k_per_image=k_per_image, pos_frac=pos_frac)
        train_sampler = None
    elif train_sampling == "global_weighted":
        train_dataset = ProposalDataset(IMAGES_DIR, train_files, labeled_train,
                                        mode="all", transform=train_transform)
        labels = [e[2] for e in train_dataset.examples]
        counts = Counter(labels)
        total = len(labels)
        class_weights = {c: total / (counts[c] + 1e-6) for c in counts}
        sample_weights = [class_weights[l] for l in labels]
        train_sampler = WeightedRandomSampler(sample_weights, num_samples=total, replacement=True)
    else:
        train_dataset = ProposalDataset(IMAGES_DIR, train_files, labeled_train,
                                        mode="all", transform=train_transform)
        train_sampler = None

    # val dataset
    if val_mode == "all":
        val_dataset = ProposalDataset(IMAGES_DIR, val_files, labeled_val,
                                      mode="all", transform=val_transform)
    elif val_mode == "sampled":
        val_dataset = ProposalDataset(IMAGES_DIR, val_files, labeled_val,
                                      mode="sampled_for_val", transform=val_transform,
                                      k_per_image=val_k_per_image)
    else:
        raise ValueError("Unknown val_mode")

    # dataloaders with prefetching
    if train_sampler is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  prefetch_factor=2, persistent_workers=(num_workers > 0))
    else:
        shuffle = (train_sampling != "all")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  prefetch_factor=2, persistent_workers=(num_workers > 0))

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory,
                            prefetch_factor=2, persistent_workers=(num_workers > 0))

    info = {
        "num_classes": num_classes,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "train_files": train_files,
        "val_files": val_files,
        "train_sampler": train_sampler
    }
    return train_loader, val_loader, info

# -----------------------
# Model
# -----------------------
class ProposalClassifier(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        # try modern weights API first (avoid deprecation warning), else fallback to pretrained flag
        try:
            # torchvision >= 0.13
            weights_enum = getattr(models, "ResNet18_Weights", None)
            if weights_enum is not None:
                weights = weights_enum.DEFAULT
                resnet = models.resnet18(weights=weights)
            else:
                resnet = models.resnet18(pretrained=pretrained)
        except Exception:
            # fallback
            resnet = models.resnet18(pretrained=pretrained)

        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.backbone = resnet
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits

# -----------------------
# Metrics & evaluation
# -----------------------
def evaluate_classification(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Dict:
    model.eval()
    total = 0
    correct = 0
    # confusion matrix counts [true][pred]
    conf = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            logits = model(images)
            preds = logits.argmax(dim=1)
            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                conf[t][p] += 1
                total += 1
                if t == p:
                    correct += 1

    acc = correct / max(1, total)
    # per-class precision / recall
    precisions = {}
    recalls = {}
    for c in range(num_classes):
        tp = conf[c][c]
        pred_pos = sum(conf[r][c] for r in range(num_classes))  # column sum
        true_pos = sum(conf[c][k] for k in range(num_classes))  # row sum
        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / true_pos if true_pos > 0 else 0.0
        precisions[c] = precision
        recalls[c] = recall

    return {
        "accuracy": acc,
        "total": total,
        "confusion": conf,
        "precision": precisions,
        "recall": recalls
    }

# -----------------------
# Plotting functions
# -----------------------
def plot_training_curves(history: Dict, save_path: str):
    """
    Plot training and validation metrics over epochs.
    
    history should contain:
        - 'epochs': list of epoch numbers
        - 'train_loss': list of training losses
        - 'train_acc': list of training accuracies
        - 'val_acc': list of validation accuracies
        - 'val_precision': dict of {class_id: [precisions per epoch]}
        - 'val_recall': dict of {class_id: [recalls per epoch]}
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    ax.plot(history['epochs'], history['train_loss'], 'b-o', label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training and Validation Accuracy
    ax = axes[0, 1]
    ax.plot(history['epochs'], history['train_acc'], 'b-o', label='Training Accuracy')
    if len(history['val_acc']) > 0:
        val_epochs = [e for e in history['epochs'] if e in history['val_epochs']]
        ax.plot(val_epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Classification Accuracy over Epochs')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Plot 3: Per-class Precision (Validation)
    ax = axes[1, 0]
    if len(history['val_precision']) > 0:
        for class_id, precisions in history['val_precision'].items():
            if len(precisions) > 0:
                val_epochs = [e for e in history['epochs'] if e in history['val_epochs']][:len(precisions)]
                ax.plot(val_epochs, precisions, '-o', label=f'Class {class_id}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Precision')
    ax.set_title('Per-Class Precision on Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    # Plot 4: Per-class Recall (Validation)
    ax = axes[1, 1]
    if len(history['val_recall']) > 0:
        for class_id, recalls in history['val_recall'].items():
            if len(recalls) > 0:
                val_epochs = [e for e in history['epochs'] if e in history['val_epochs']][:len(recalls)]
                ax.plot(val_epochs, recalls, '-s', label=f'Class {class_id}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Recall')
    ax.set_title('Per-Class Recall on Validation Set')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()

# -----------------------
# Training loop
# -----------------------
def train_and_evaluate(num_epochs: int = NUM_EPOCHS):
    # build dataloaders
    train_loader, val_loader, info = build_dataloaders(
        train_sampling=TRAIN_SAMPLING,
        k_per_image=K_PER_IMAGE,
        pos_frac=POS_FRAC,
        val_mode="all" if VAL_MODE == "all" else "sampled",
        val_k_per_image=VAL_K_PER_IMAGE,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        debug_limit_train=DEBUG_LIMIT_TRAIN_IMAGES,
        debug_limit_val=DEBUG_LIMIT_VAL_IMAGES
    )

    num_classes = info["num_classes"]
    print("Num classes:", num_classes)
    print("Train examples (proposals):", info["train_examples"])
    print("Val examples (proposals):", info["val_examples"])

    model = ProposalClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)

    # class-weighted loss if requested
    if USE_CLASS_WEIGHTED_LOSS:
        labels = [e[2] for e in info["train_dataset"].examples]
        counts = Counter(labels)
        total = sum(counts.values())
        # weight = total / (count) gives inverse frequency; normalize to mean 1
        raw_weights = [ (total / (counts.get(c,1) + 1e-6)) for c in range(num_classes) ]
        mean_w = float(sum(raw_weights)) / len(raw_weights)
        weights = [w / mean_w for w in raw_weights]
        weight_tensor = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using class-weighted CrossEntropyLoss with weights:", weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Mixed precision scaler
    scaler = GradScaler() if USE_MIXED_PRECISION and DEVICE == "cuda" else None
    if scaler:
        print("Using mixed precision training (FP16)")

    best_val_acc = -1.0
    
    # History for plotting
    history = {
        'epochs': [],
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_epochs': [],
        'val_precision': defaultdict(list),
        'val_recall': defaultdict(list)
    }

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            
            if scaler:
                # Mixed precision training
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

            if batch_idx % PRINT_EVERY_BATCHES == 0:
                avg_loss = running_loss / max(1, running_total)
                avg_acc = running_correct / max(1, running_total)
                print(f"[Epoch {epoch}] Batch {batch_idx} | avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f} (seen {running_total} samples)")

        # epoch end training metrics
        epoch_loss = running_loss / max(1, running_total)
        epoch_acc = running_correct / max(1, running_total)
        print(f"Epoch {epoch} training complete. loss={epoch_loss:.4f} acc={epoch_acc:.4f}")
        
        # Store training metrics
        history['epochs'].append(epoch)
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Evaluate on validation set (based on VAL_EVERY_N_EPOCHS)
        if epoch % VAL_EVERY_N_EPOCHS == 0 or epoch == num_epochs:
            print("Evaluating on validation set...")
            val_stats = evaluate_classification(model, val_loader, DEVICE, num_classes)
            print(f"Validation: total={val_stats['total']} accuracy={val_stats['accuracy']:.4f}")
            print("Per-class precision / recall:")
            for c in range(num_classes):
                print(f"  class {c}: prec={val_stats['precision'][c]:.4f} recall={val_stats['recall'][c]:.4f}")

            # print condensed confusion counts for classes 0..N-1
            print("Confusion matrix (rows=true, cols=pred):")
            for r in range(num_classes):
                print("  " + " ".join(str(x) for x in val_stats["confusion"][r]))
            
            # Store validation metrics
            history['val_acc'].append(val_stats['accuracy'])
            history['val_epochs'].append(epoch)
            for c in range(num_classes):
                history['val_precision'][c].append(val_stats['precision'][c])
                history['val_recall'][c].append(val_stats['recall'][c])

            # checkpointing
            ckpt_path = os.path.join(CKPT_DIR, f"epoch{epoch:02d}_valacc{val_stats['accuracy']:.4f}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_stats": val_stats,
                "train_info": info,
                "history": history
            }, ckpt_path)
            print("Saved checkpoint to:", ckpt_path)

            # save best model separately
            if val_stats["accuracy"] > best_val_acc:
                best_val_acc = val_stats["accuracy"]
                best_path = os.path.join(CKPT_DIR, "best_model.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_stats": val_stats,
                    "train_info": info,
                    "history": history
                }, best_path)
                print(f"Saved new best model to: {best_path} (val_acc={best_val_acc:.4f})")

    # Plot training curves at the end
    plot_path = os.path.join(CKPT_DIR, "training_curves.png")
    plot_training_curves(history, plot_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Training curves saved to: {plot_path}")
    print("="*60)

if __name__ == "__main__":
    train_and_evaluate(NUM_EPOCHS)