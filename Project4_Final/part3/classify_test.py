import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import json
from classifier import ProposalClassifier
import torchvision.transforms as T


IMG_SIZE = 224
DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNO_DIR   = os.path.join(DATA_ROOT, "annotations")

END_PATH = "~/courses/02516/IDLCV/Project4_Final/results"
os.makedirs(END_PATH, exist_ok=True)

SPLITS_PATH = os.path.join(END_PATH, "splits.json")

PROPOSALS_TEST = os.path.join(
    "~/courses/02516/IDLCV/Project4_Final/part1", "proposals_edgeboxes_test.json"
)




# Load the trained model (replace 'model.pth' with your file)
model = ProposalClassifier()
model.load_state_dict(torch.load('~/courses/02516/best_model.pth'))
model.eval()

# Preprocessing transform (adjust to match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# transforms
train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomApply([T.ColorJitter(0.2,0.2,0.2,0.05)], p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

with open(PROPOSALS_TEST, 'r') as f:
    data = json.load(f)

print(list(data.keys()))
# Your proposals dictionary
proposals = {
    "potholes312.png": [[221, 5, 289, 50], [266, 9, 306, 26]] 
}

# # Dictionary to store scores: {image: [(box, score)]}
# detections = {}

# for img_name, boxes in proposals.items():
#     # Load the image (assume images are in 'test_images/' directory)
#     img = Image.open(f'test_images/{img_name}').convert('RGB')
#     img_tensor = transform(img)  # But we need patches, not full image
    
#     scores = []
#     for box in boxes:
#         x1, y1, x2, y2 = box
#         patch = img.crop((x1, y1, x2, y2))  # Crop the proposal
#         patch_tensor = transform(patch).unsqueeze(0)  # Add batch dim
#         with torch.no_grad():
#             score = model(patch_tensor).item()  # Confidence score (0-1)
#         scores.append((box, score))
    
#     detections[img_name] = scores