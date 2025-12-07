import numpy as np

def iou(boxA, boxB):
    # box = [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea)
    return iou_val

def nms(detections_per_image, iou_threshold=0.5, score_threshold=0.5):
    # detections_per_image = [(box, score), ...]
    # Sort by score descending
    detections_per_image = sorted([d for d in detections_per_image if d[1] > score_threshold], key=lambda x: x[1], reverse=True)
    
    keep = []
    while detections_per_image:
        best = detections_per_image.pop(0)
        keep.append(best)
        detections_per_image = [d for d in detections_per_image if iou(best[0], d[0]) < iou_threshold]
    
    return keep

def compute_ap(recall, precision):
    # Interpolate and compute area (11-point or all-point)
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def compute_ap_11point(recall, precision):
    # 11-point AP (at recall levels 0.0, 0.1, ..., 1.0)
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0
    return ap

def evaluate_detections(final_detections, ground_truth, iou_threshold=0.5):
    all_tp = []
    all_scores = []
    total_gt = 0
    
    for img_name in final_detections:
        dets = sorted(final_detections[img_name], key=lambda x: x[1], reverse=True)
        gts = ground_truth.get(img_name, [])
        total_gt += len(gts)
        
        matched_gt = [False] * len(gts)
        for det_box, score in dets:
            is_tp = False
            for j, gt_box in enumerate(gts):
                if not matched_gt[j] and iou(det_box, gt_box) >= iou_threshold:
                    is_tp = True
                    matched_gt[j] = True
                    break
            all_tp.append(is_tp)
            all_scores.append(score)
    
    # Sort by score descending (across all images)
    indices = np.argsort(-np.array(all_scores))
    all_tp = np.array(all_tp)[indices]
    all_scores = np.array(all_scores)[indices]
    
    # Cumulative TP and precision/recall
    cum_tp = np.cumsum(all_tp)
    recall = cum_tp / total_gt
    precision = cum_tp / np.arange(1, len(cum_tp) + 1)
    
    # ap = compute_ap(recall, precision)
    ap = compute_ap_11point(recall, precision)
    return ap

