import os
import json
import numpy as np
import xml.etree.ElementTree as ET

# --- paths ---
DATA_ROOT = "/dtu/datasets1/02516/potholes"
ANNO_DIR = os.path.join(DATA_ROOT, "annotations")

END_PATH = "/zhome/4d/5/147570/IDLCV/Project_4/results"

SPLITS_PATH = os.path.join(END_PATH, "splits.json")
PROPOSALS_TRAIN_PATH = os.path.join(
    END_PATH, "proposals_edgeboxes", "proposals_edgeboxes_train.json"
)

OUTPUT_PATH = os.path.join(END_PATH, "labeled_proposals_train.json")

# --- IoU helper ---
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - interArea

    if union <= 0:
        return 0.0
    return interArea / union

def load_gt_boxes(img_fname):
    base = os.path.splitext(img_fname)[0]
    xml_path = os.path.join(ANNO_DIR, base + ".xml")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        b = obj.find("bndbox")
        xmin = float(b.find("xmin").text)
        ymin = float(b.find("ymin").text)
        xmax = float(b.find("xmax").text)
        ymax = float(b.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return np.array(boxes)

# --- main labeling ---
with open(SPLITS_PATH, "r") as f:
    splits = json.load(f)
train_files = splits["train"]

with open(PROPOSALS_TRAIN_PATH, "r") as f:
    proposals = json.load(f)

labeled = {}

for fname in train_files:
    gt = load_gt_boxes(fname)
    props = np.array(proposals[fname])

    labels = []
    for p in props:
        max_iou = 0.0
        for g in gt:
            max_iou = max(max_iou, iou(p, g))

        label = 1 if max_iou >= 0.5 else 0
        labels.append({"box": p.tolist(), "label": int(label), "iou": float(max_iou)})

    labeled[fname] = labels

with open(OUTPUT_PATH, "w") as f:
    json.dump(labeled, f, indent=2)

print("Saved labeled proposals to:", OUTPUT_PATH)
