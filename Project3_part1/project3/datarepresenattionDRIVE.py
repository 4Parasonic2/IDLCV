import os
import numpy as np
import glob
import PIL.Image as Image

import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# use dataset folder relative to this file (or override with env var)
DATASET_ROOT = os.environ.get(
    'DRIVE_DATASET_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataDRIVE'))
)

# use os.path.join for subpaths (no trailing slashes)
TRAIN_IMAGES_PATH = os.path.join('training', 'images')
TRAIN_MASKS_PATH  = os.path.join('training', 'mask')
TEST_IMAGES_PATH  = os.path.join('test', 'images')
TEST_MASKS_PATH   = os.path.join('test', 'mask')

# quick sanity check (prints warnings if folders are missing)
for name, rel in (
    ("TRAIN_IMAGES", TRAIN_IMAGES_PATH),
    ("TRAIN_MASKS", TRAIN_MASKS_PATH),
    ("TEST_IMAGES", TEST_IMAGES_PATH),
    ("TEST_MASKS", TEST_MASKS_PATH),
):
    full = os.path.join(DATASET_ROOT, rel)
    if not os.path.isdir(full):
        print(f"Warning: {name} directory not found: {full}")

# =============================================================================
# DATASET STATISTICS
# =============================================================================

def count_files():
    """
    Count files in DRIVE dataset
    """
    # Find all files
    train_images = glob.glob(os.path.join(DATASET_ROOT, TRAIN_IMAGES_PATH, '*.tif'))
    train_masks = glob.glob(os.path.join(DATASET_ROOT, TRAIN_MASKS_PATH, '*.gif'))
    test_images = glob.glob(os.path.join(DATASET_ROOT, TEST_IMAGES_PATH, '*.tif'))
    test_masks = glob.glob(os.path.join(DATASET_ROOT, TEST_MASKS_PATH, '*.gif'))
    
    print("="*60)
    print("DRIVE DATASET - FILE COUNTS")
    print("="*60)
    
    print(f"\nTRAINING:")
    print(f"  .tif files: {len(train_images)}")
    print(f"  .gif files: {len(train_masks)}")
    
    print(f"\nTEST:")
    print(f"  .tif files: {len(test_images)}")
    print(f"  .gif files: {len(test_masks)}")
    
    print(f"\nTOTAL:")
    print(f"  .tif files: {len(train_images) + len(test_images)}")
    print(f"  .gif files: {len(train_masks) + len(test_masks)}")
    
    print("="*60)
    
    return train_images, train_masks, test_images, test_masks

# =============================================================================
# DATASET CLASS
# =============================================================================

class DRIVEDataset(datasets.VisionDataset):
    def __init__(self, root='dataDRIVE', train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.train = train
        
        if self.train:
            img_path = os.path.join(root, TRAIN_IMAGES_PATH)
            mask_path = os.path.join(root, TRAIN_MASKS_PATH)
        else:
            img_path = os.path.join(root, TEST_IMAGES_PATH)
            mask_path = os.path.join(root, TEST_MASKS_PATH)
        
        self.images = sorted(glob.glob(os.path.join(img_path, '*.tif')))
        self.masks = sorted(glob.glob(os.path.join(mask_path, '*.gif')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)  # ensures same resizing
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        return image, mask

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Count files
    count_files()
    
    # Create datasets
    print("\nCreating PyTorch datasets...")
    
    size = 128
    train_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    
    batch_size = 6
    
    trainset = DRIVEDataset(root=DATASET_ROOT, train=True, transform=train_transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    
    testset = DRIVEDataset(root=DATASET_ROOT, train=False, transform=test_transform)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)
    
    print(f"\nLoaded {len(trainset)} training images")
    print(f"Loaded {len(testset)} test images")