#!/usr/bin/env python3
"""
Evaluate EdgeBoxes proposals (Task 3) and plot recall curves:

- Loads train split and EdgeBoxes proposals
- Loads Pascal VOC-style annotations
- Computes recall vs number of proposals and IoU thresholds
- Saves a plot: proposal_recall_edgeboxes.png
"""

import os
import json
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt   # <--- added

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

DATA_ROOT = "/dtu/datasets1/02516/potholes"
IMAGES_DIR = os.path.join(DATA_ROOT, "images")
ANNO_DIR   = os.path.join(DATA_ROOT, "annotations")

END_PATH = "/zhome/4d/5/147570/IDLCV/Project_4/results"
os.makedirs(END_PATH, exist_ok=True)

SPLITS_PATH = os.path.join(END_PATH, "splits.json")

PROPOSALS_TRAIN = os.path.join(
    END_PATH, "proposals_edgeboxes", "proposals_edgeboxes_train.json"
)

# IoU thresholds and number of proposals to test
IOU_THRESHOLDS = [0.3, 0.5, 0.7]
NUM_PROPOSALS_LIST = [50, 100, 200, 500, 1000, 2000]

PLOT_PATH = os.path.join(END_PATH, "proposal_recall_edgeboxes.png")

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def img_to_anno_path(img_filename):
    base, _ = os.path.splitext(img_filename)
    return os.path.join(ANNO_DIR, base + ".xml")


def load_gt_boxes(img_filename):
    anno_path = img_to_anno_path(img_filename)
    if not os.path.isfile(anno_path):
        print(f"WARNING: annotation not found for {img_filename}: {anno_path}")
        return np.zeros((0, 4), dtype=np.float32)

    tree = ET.parse(anno_path)
    root = tree.getroot()

    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(boxes, dtype=np.float32)


def iou_matrix(boxes1, boxes2):
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(b1[..., 0], b2[..., 0])
    inter_y1 = np.maximum(b1[..., 1], b2[..., 1])
    inter_x2 = np.minimum(b1[..., 2], b2[..., 2])
    inter_y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None)
    inter_h = np.clip(inter_y2 - inter_y1, a_min=0, a_max=None)
    inter_area = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    area1 = area1[:, None]
    area2 = area2[None, :]

    union = area1 + area2 - inter_area
    union = np.clip(union, a_min=1e-6, a_max=None)

    iou = inter_area / union
    return iou


def evaluate_recall(train_files, proposals_dict):
    results = {
        iou_thr: {
            "total_gt": 0,
            "covered_at_N": {N: 0 for N in NUM_PROPOSALS_LIST}
        }
        for iou_thr in IOU_THRESHOLDS
    }

    for img_fname in train_files:
        gt_boxes = load_gt_boxes(img_fname)
        if gt_boxes.size == 0:
            continue

        prop_boxes = np.array(proposals_dict.get(img_fname, []), dtype=np.float32)
        if prop_boxes.size == 0:
            for iou_thr in IOU_THRESHOLDS:
                results[iou_thr]["total_gt"] += gt_boxes.shape[0]
            continue

        for iou_thr in IOU_THRESHOLDS:
            total_gt_img = gt_boxes.shape[0]
            for N in NUM_PROPOSALS_LIST:
                top_props = prop_boxes[:N]
                ious = iou_matrix(gt_boxes, top_props)
                max_iou_per_gt = ious.max(axis=1) if ious.size > 0 else np.zeros((total_gt_img,))
                covered = (max_iou_per_gt >= iou_thr).sum()

                results[iou_thr]["covered_at_N"][N] += int(covered)

            results[iou_thr]["total_gt"] += total_gt_img

    recall = {
        iou_thr: {
            N: results[iou_thr]["covered_at_N"][N] / max(results[iou_thr]["total_gt"], 1)
            for N in NUM_PROPOSALS_LIST
        }
        for iou_thr in IOU_THRESHOLDS
    }

    return recall, results


def plot_recall(recall, save_path):
    """
    Make a nice recall vs N plot and save as PNG.
    """
    plt.figure(figsize=(6, 4))

    for iou_thr in IOU_THRESHOLDS:
        recalls = [recall[iou_thr][N] for N in NUM_PROPOSALS_LIST]
        # plot as percentage
        recalls_pct = [r * 100.0 for r in recalls]
        plt.plot(NUM_PROPOSALS_LIST, recalls_pct, marker="o", label=f"IoU ≥ {iou_thr}")

    plt.xlabel("Number of proposals per image")
    plt.ylabel("Recall of ground-truth boxes [%]")
    plt.title("EdgeBoxes proposal recall on training set")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved plot to {save_path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    # Load splits
    with open(SPLITS_PATH, "r") as f:
        splits = json.load(f)
    train_files = splits.get("train", [])
    if not train_files:
        raise RuntimeError("No 'train' split found in splits.json")

    # Load proposals for train
    with open(PROPOSALS_TRAIN, "r") as f:
        proposals_dict = json.load(f)

    recall, raw = evaluate_recall(train_files, proposals_dict)

    print("==== Proposal Recall (EdgeBoxes) on TRAIN set ====")
    print(f"Total GT boxes: {next(iter(raw.values()))['total_gt']}")
    print(f"Tested Ns: {NUM_PROPOSALS_LIST}")
    print()

    for iou_thr in IOU_THRESHOLDS:
        print(f"IoU threshold = {iou_thr}")
        for N in NUM_PROPOSALS_LIST:
            r = recall[iou_thr][N]
            print(f"  N={N:4d}: recall = {r*100:5.1f}%")
        print()

    # Optional suggestion
    target_iou = 0.5
    target_recall = 0.9
    best_N = None
    for N in NUM_PROPOSALS_LIST:
        if recall[target_iou][N] >= target_recall:
            best_N = N
            break

    if best_N is not None:
        print(f"Suggested N (IoU={target_iou}, recall>={target_recall*100:.0f}%): {best_N}")
    else:
        print(f"No N in {NUM_PROPOSALS_LIST} reached {target_recall*100:.0f}% at IoU={target_iou}")

    # ---- NEW: make and save plot ----
    plot_recall(recall, PLOT_PATH)


if __name__ == "__main__":
    main()
