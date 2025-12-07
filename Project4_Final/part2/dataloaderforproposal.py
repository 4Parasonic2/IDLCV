#!/usr/bin/env python3
"""
proposal_dataloader.py

Provides:
- ProposalDataset (crops/resizes proposal boxes from images)
- build_dataloaders(...) function that returns (train_loader, val_loader, info_dict)
  where info_dict contains num_classes, train_dataset, val_dataset, and optionally sampler info.

Sampling strategies:
- "per_image": sample up to k_per_image proposals per image with pos_frac positives
- "global_weighted": use WeightedRandomSampler over proposals with inverse-freq weights
- "none": return DataLoader over all proposals (shuffled)
"""

import os
import json
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional

from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# -----------------------
# Default config (you can override by passing args to build_dataloaders)
# -----------------------
DEFAULT_IMG_SIZE = 224

# -----------------------
# Helpers
# -----------------------
def _box_to_int_coords(box):
    # box may be list of floats or strings
    return [int(round(float(x))) for x in box]

# -----------------------
# Dataset
# -----------------------
class ProposalDataset(Dataset):
    """
    ProposalDataset returns (patch_tensor, label) for each labeled proposal.

    Args:
      images_dir: directory with image files
      image_list: list of image filenames (relative paths used as keys in labeled_proposals)
      labeled_proposals: dict mapping image filename -> list of dicts {"box": [...], "label": int, "iou": float}
      mode: "all" or "per_image_sample"
      transform: torchvision transform applied to cropped patch
      k_per_image / pos_frac: used for per_image_sample mode
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

        self.examples = []  # list of tuples (fname, box, label)

        if self.mode == "all":
            for fname in self.image_list:
                props = labeled_proposals.get(fname, [])
                for p in props:
                    self.examples.append((fname, p["box"], int(p["label"])))
        elif self.mode == "per_image_sample":
            local_rng = random.Random(42)
            for fname in self.image_list:
                props = labeled_proposals.get(fname, [])
                if not props:
                    continue
                positives = [p for p in props if int(p["label"]) != 0]
                negatives = [p for p in props if int(p["label"]) == 0]

                num_pos = min(len(positives), int(round(self.k_per_image * self.pos_frac)))
                num_neg = min(len(negatives), self.k_per_image - num_pos)

                sampled_pos = local_rng.sample(positives, num_pos) if num_pos > 0 else []
                sampled_neg = local_rng.sample(negatives, num_neg) if num_neg > 0 else []

                # add to examples
                for p in (sampled_pos + sampled_neg):
                    self.examples.append((fname, p["box"], int(p["label"])))
        else:
            raise ValueError("mode must be 'all' or 'per_image_sample'")

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
                # degenerate box -> center small crop
                cx, cy = img.width // 2, img.height // 2
                half = max(1, min(img.width, img.height) // 8)
                crop = img.crop((cx-half, cy-half, cx+half, cy+half))
            else:
                crop = img.crop((xmin, ymin, xmax, ymax))

            if self.transform:
                patch = self.transform(crop)
            else:
                # default transform: resize -> tensor -> normalize
                t = T.Compose([
                    T.Resize((DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
                patch = t(crop)
        return patch, label

# -----------------------
# Build dataloaders
# -----------------------
def build_dataloaders(images_dir: str,
                      splits_path: str,
                      labeled_proposals_path: str,
                      labeled_proposals_val_path: Optional[str] = None,
                      img_size: int = DEFAULT_IMG_SIZE,
                      batch_size: int = 32,
                      num_workers: int = 4,
                      pin_memory: bool = True,
                      sampling_strategy: str = "per_image",
                      k_per_image: int = 8,
                      pos_frac: float = 0.5,
                      val_fraction: float = 0.10,
                      debug_limit_images: Optional[int] = None
                      ) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Returns: train_loader, val_loader, info_dict

    info_dict contains:
      - num_classes
      - train_dataset, val_dataset
      - train_examples_count, val_examples_count
      - sampler (if global_weighted used) else None
    """
    # load splits
    with open(splits_path, "r") as f:
        splits = json.load(f)
    train_files = splits.get("train", [])
    val_files = splits.get("val", []) or splits.get("validation", []) or []

    # create val split if missing
    if not val_files:
        n_val = max(1, int(val_fraction * len(train_files)))
        rnd = random.Random(42)
        shuffled = train_files.copy()
        rnd.shuffle(shuffled)
        val_files = shuffled[:n_val]
        train_files = shuffled[n_val:]

    # debug limit
    if debug_limit_images is not None:
        train_files = train_files[:debug_limit_images]
        val_files = val_files[:max(1, debug_limit_images//5)]

    # load labeled proposals
    with open(labeled_proposals_path, "r") as f:
        labeled_train = json.load(f)

    labeled_val = {}
    if labeled_proposals_val_path and os.path.exists(labeled_proposals_val_path):
        with open(labeled_proposals_val_path, "r") as f:
            labeled_val = json.load(f)
    else:
        for fname in val_files:
            if fname in labeled_train:
                labeled_val[fname] = labeled_train[fname]

        # fallback: if labeled_val empty, sample some labeled images
        if len(labeled_val) == 0:
            available = list(labeled_train.keys())
            if len(available) > 0:
                sample_n = min(len(available), max(1, int(val_fraction * len(available))))
                rnd = random.Random(123)
                sampled = rnd.sample(available, sample_n)
                for fname in sampled:
                    labeled_val[fname] = labeled_train[fname]
                # ensure val_files contains them and exclude from train_files
                val_files = list(set(val_files + sampled))
                train_files = [f for f in train_files if f not in sampled]

    # infer num_classes from labels present
    all_labels = set()
    for v in labeled_train.values():
        for p in v:
            all_labels.add(int(p["label"]))
    all_labels.add(0)
    num_classes = max(all_labels) + 1

    # transforms (train/val)
    train_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.05)], p=0.5),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # datasets
    train_mode = "per_image_sample" if sampling_strategy == "per_image" else "all"
    train_dataset = ProposalDataset(images_dir, train_files, labeled_train,
                                    mode=train_mode, transform=train_transform,
                                    k_per_image=k_per_image, pos_frac=pos_frac)
    val_dataset = ProposalDataset(images_dir, val_files, labeled_val,
                                  mode="all", transform=val_transform)

    # build dataloaders / sampler
    sampler = None
    if sampling_strategy == "global_weighted":
        labels = [e[2] for e in train_dataset.examples]
        counts = Counter(labels)
        num_samples = len(labels)
        class_weights = {cls: num_samples / (count + 1e-6) for cls, count in counts.items()}
        sample_weights = [class_weights[l] for l in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                                  num_workers=num_workers, pin_memory=pin_memory)
    else:
        shuffle = (sampling_strategy == "none")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    info = {
        "num_classes": num_classes,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "train_examples": len(train_dataset),
        "val_examples": len(val_dataset),
        "sampler": sampler
    }

    return train_loader, val_loader, info
