import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

class PH2WeakDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, clicks_dir, transform=None):
        self.root = root_dir
        self.transform = transform
        self.clicks_dir = clicks_dir
        self.cases = sorted([d for d in os.listdir(root_dir) if d.startswith("IMD")])

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.root, case)

        # ---- Image ----
        img_path = os.path.join(case_dir, f"{case}_Dermoscopic_Image", f"{case}.bmp")
        img = Image.open(img_path).convert("RGB")

        # ---- Full mask (only for eval) ----
        mask_path = os.path.join(case_dir, f"{case}_lesion", f"{case}_lesion.bmp")
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 128).astype(np.float32)
        mask = torch.tensor(mask).unsqueeze(0)

        # ---- Clicks ----
        pos_path = os.path.join(self.clicks_dir, f"{case}_pos.npy")
        neg_path = os.path.join(self.clicks_dir, f"{case}_neg.npy")

        pos = np.load(pos_path)
        neg = np.load(neg_path)

        # ---- Apply transform ----
        if self.transform:
            img = self.transform(img)
            mask = T.functional.resize(mask, img.shape[1:], antialias=True)

        return img, mask, pos, neg
