import os
import glob
import PIL.Image as Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

# use dataset folder relative to this file (or override with env var)
DATASET_ROOT = os.environ.get(
    'PH2_DATASET_ROOT',
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataPH2'))
)

# quick sanity check
if not os.path.isdir(DATASET_ROOT):
    print(f"Warning: dataset root not found: {DATASET_ROOT}")

# =============================================================================
# DATASET STATISTICS
# =============================================================================

def count_files():
    """
    Count image, lesion, and ROI files in PH2 dataset.
    """
    all_patients = sorted(glob.glob(os.path.join(DATASET_ROOT, 'IMD*')))
    
    dermoscopic_count = 0
    lesion_count = 0
    roi_count = 0

    for patient in all_patients:
        dermoscopic = glob.glob(os.path.join(patient, '*_Dermoscopic_Image', '*.bmp'))
        lesion = glob.glob(os.path.join(patient, '*_lesion', '*.bmp'))
        roi = glob.glob(os.path.join(patient, '*_roi', '*.bmp'))

        dermoscopic_count += len(dermoscopic)
        lesion_count += len(lesion)
        roi_count += len(roi)

    print("=" * 60)
    print("PH2 DATASET - FILE COUNTS")
    print("=" * 60)
    print(f"Total patient folders : {len(all_patients)}")
    print(f"Dermoscopic images (.bmp): {dermoscopic_count}")
    print(f"Lesion masks (.bmp):       {lesion_count}")
    print(f"ROI masks (.bmp):          {roi_count}")
    print("=" * 60)

    return all_patients

# =============================================================================
# DATASET CLASS
# =============================================================================

class PH2Dataset(datasets.VisionDataset):
    """
    PyTorch dataset loader for the PH² dataset.
    Each sample folder (IMDxxx) contains:
      - IMDxxx_Dermoscopic_Image/*.bmp  -> RGB image
      - IMDxxx_lesion/*.bmp             -> lesion mask
      - IMDxxx_roi/*.bmp (optional)     -> region of interest mask
    """
    def __init__(self, root='dataPH2', transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = sorted(glob.glob(os.path.join(root, 'IMD*')))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        folder = self.samples[idx]

        # Dermoscopic image (required)
        img_files = glob.glob(os.path.join(folder, '*_Dermoscopic_Image', '*.bmp'))
        lesion_files = glob.glob(os.path.join(folder, '*_lesion', '*.bmp'))
        roi_files = glob.glob(os.path.join(folder, '*_roi', '*.bmp'))

        if len(img_files) == 0 or len(lesion_files) == 0:
            raise FileNotFoundError(f"Missing required files in folder: {folder}")

        img_path = img_files[0]
        lesion_path = lesion_files[0]
        roi_path = roi_files[0] if len(roi_files) > 0 else None

        image = Image.open(img_path).convert('RGB')
        lesion = Image.open(lesion_path).convert('L')

        # If ROI mask missing → create blank image of same size
        if roi_path is not None:
            roi = Image.open(roi_path).convert('L')
        else:
            roi = Image.new('L', lesion.size, color=0)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            lesion = self.target_transform(lesion)
            roi = self.target_transform(roi)
        else:
            lesion = transforms.ToTensor()(lesion)
            roi = transforms.ToTensor()(roi)

        return image, lesion, roi

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

    # target_transform should resize and convert masks to tensors as well
    target_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])

    # Load full dataset
    # ensure masks (lesion, roi) are resized and converted to tensors via target_transform
    full_dataset = PH2Dataset(root=DATASET_ROOT, transform=train_transform, target_transform=target_transform)

    # Split 80/20 for train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    trainset, testset = random_split(full_dataset, [train_size, test_size])

    # Data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

    print(f"\nLoaded {len(trainset)} training samples")
    print(f"Loaded {len(testset)} test samples")

    # Optional: check batch
    for batch in train_loader:
        images, lesions, rois = batch
        print(f"Batch shapes -> images: {images.shape}, lesions: {lesions.shape}, rois: {rois.shape}")
        break
