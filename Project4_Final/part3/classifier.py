import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import os
from typing import List, Dict, Tuple, Optional

DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
END_PATH = "/zhome/99/4/213789/IDLCV/finalprojecy"
SPLITS_PATH = os.path.join(END_PATH, "splits.json")
LABELED_PROPOSALS_PATH = os.path.join(END_PATH, "labeled_proposals_train.json")
LABELED_PROPOSALS_VAL = os.path.join(END_PATH, "labeled_proposals_val.json")  # optional, may not exist
from PIL import Image

IMG_SIZE = 224

import random 
SEED = 42



def _box_to_int_coords(box):
    return [int(round(float(x))) for x in box]


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