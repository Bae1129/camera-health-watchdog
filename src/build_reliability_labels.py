import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def bbox_iou_xywh(box1, box2):
    """
    IoU between two COCO-format bounding boxes:
    [x, y, width, height]
    """

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)

    intersection = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_coco_gt(annotation_path):
    """
    Load COCO annotations and group GT objects by image_id.
    Crowd annotations are excluded.
    """

    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    gt_by_image = defaultdict(list)

    for ann in coco["annotations"]:

        if ann.get("iscrowd", 0) == 1:
            continue

        gt_by_image[ann["image_id"]].append({
            "ann_id": ann["id"],
            "category_id": ann["category_id"],
            "bbox": ann["bbox"],
        })

    return gt_by_image


def load_predictions(prediction_path):
    """
    Load detector predictions in COCO result format.
    """

    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)

    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)

    return pred_by_image


def match_predictions_to_gt(
    gt_list,
    pred_list,
    conf_thr,
    iou_thr,
):
    """
    Greedy one-to-one matching.

    Conditions:
    1. prediction confidence >= conf_thr
    2. same category
    3. IoU >= iou_thr

    Returns:
        {
            GT annotation ID: {
                "score": ...,
                "iou": ...,
                "bbox": ...
            }
        }
    """

    predictions = [
        p for p in pred_list
        if p["score"] >= conf_thr
    ]

    predictions = sorted(
        predictions,
        key=lambda x: x["score"],
        reverse=True,
    )

    matched_gt_ids = set()
    matches = {}

    for pred in predictions:

        best_gt = None
        best_iou = 0.0

        for gt in gt_list:

            ann_id = gt["ann_id"]

            if ann_id in matched_gt_ids:
                continue

            if pred["category_id"] != gt["category_id"]:
                continue

            iou = bbox_iou_xywh(
                pred["bbox"],
                gt["bbox"],
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = gt

        if best_gt is not None and best_iou >= iou_thr:

            ann_id = best_gt["ann_id"]

            matched_gt_ids.add(ann_id)

            matches[ann_id] = {
                "score": float(pred["score"]),
                "iou": float(best_iou),
                "bbox": pred["bbox"],
            }

    return matches


def safe_mean(values):
    if not values:
        return None
    return mean(values)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ann",
        required=True,
        help="COCO ground-truth annotation JSON",
    )

    parser.add_argument(
        "--clean-pred",
        required=True,
        help="Clean detector prediction JSON",
    )

    parser.add_argument(
        "--degraded-pred",
        required=True,
        help="Degraded detector prediction JSON",
    )

    parser.add_argument(
        "--condition",
        required=True,
        help="Example: defocus, motion_blur, gaussian_noise",
    )

    parser.add_argument(
        "--severity",
        required=True,
        help="Example: d11, sigma15",
    )

    parser.add_argument(
        "--object-output",
        required=True,
        help="Object-level output CSV",
    )

    parser.add_argument(
        "--frame-output",
        required=True,
        help="Frame-level summary CSV",
    )

    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--iou-thr",
        type=float,
        default=0.50,
    )

    args = parser.parse_args()

    gt_by_image = load_coco_gt(args.ann)

    clean_pred_by_image = load_predictions(
        args.clean_pred
    )

    degraded_pred_by_image = load_predictions(
        args.degraded_pred
    )

    object_rows = []
    frame_rows = []

    for image_id, gt_list in gt_by_image.items():

        clean_matches = match_predictions_to_gt(
            gt_list,
            clean_pred_by_image.get(image_id, []),
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
        )

        degraded_matches = match_predictions_to_gt(
            gt_list,
            degraded_pred_by_image.get(image_id, []),
            conf_thr=args.conf_thr,
            iou_thr=args.iou_thr,
        )

        clean_reference_ids = set(clean_matches.keys())

        degraded_ids = set(degraded_matches.keys())

        retained_ids = (
            clean_reference_ids
            & degraded_ids
        )

        # -------------------------------------------------
        # Object-level labels
        # -------------------------------------------------

        for gt in gt_list:

            ann_id = gt["ann_id"]

            clean = clean_matches.get(ann_id)
            degraded = degraded_matches.get(ann_id)

            clean_detected = clean is not None
            degraded_detected = degraded is not None

            retained_from_clean = (
                clean_detected
                and degraded_detected
            )

            clean_score = (
                clean["score"]
                if clean_detected
                else None
            )

            clean_iou = (
                clean["iou"]
                if clean_detected
                else None
            )

            # Clean reference 객체인데 열화 후 miss라면
            # degraded score/IoU = 0으로 취급
            if clean_detected:

                degraded_score = (
                    degraded["score"]
                    if degraded_detected
                    else 0.0
                )

                degraded_iou = (
                    degraded["iou"]
                    if degraded_detected
                    else 0.0
                )

                score_ratio = (
                    degraded_score / clean_score
                    if clean_score > 0
                    else None
                )

                iou_ratio = (
                    degraded_iou / clean_iou
                    if clean_iou > 0
                    else None
                )

            else:

                degraded_score = (
                    degraded["score"]
                    if degraded_detected
                    else None
                )

                degraded_iou = (
                    degraded["iou"]
                    if degraded_detected
                    else None
                )

                score_ratio = None
                iou_ratio = None

            object_rows.append({
                "image_id": image_id,
                "ann_id": ann_id,
                "category_id": gt["category_id"],

                "condition": args.condition,
                "severity": args.severity,

                "clean_detected": int(clean_detected),
                "degraded_detected": int(degraded_detected),
                "retained_from_clean": int(retained_from_clean),

                "clean_score": clean_score,
                "degraded_score": degraded_score,

                "clean_iou": clean_iou,
                "degraded_iou": degraded_iou,

                "score_ratio": score_ratio,
                "iou_ratio": iou_ratio,
            })

        # -------------------------------------------------
        # Frame-level summary
        # -------------------------------------------------

        K = len(clean_reference_ids)
        k = len(retained_ids)

        if K > 0:

            detection_retention = k / K

            clean_scores = [
                clean_matches[ann_id]["score"]
                for ann_id in clean_reference_ids
            ]

            degraded_scores = [
                degraded_matches[ann_id]["score"]
                if ann_id in degraded_matches
                else 0.0
                for ann_id in clean_reference_ids
            ]

            clean_ious = [
                clean_matches[ann_id]["iou"]
                for ann_id in clean_reference_ids
            ]

            degraded_ious = [
                degraded_matches[ann_id]["iou"]
                if ann_id in degraded_matches
                else 0.0
                for ann_id in clean_reference_ids
            ]

        else:

            detection_retention = None

            clean_scores = []
            degraded_scores = []

            clean_ious = []
            degraded_ious = []

        frame_rows.append({
            "image_id": image_id,

            "condition": args.condition,
            "severity": args.severity,

            "total_gt": len(gt_list),

            "clean_reference_count": K,
            "degraded_detected_count": len(degraded_ids),
            "retained_count": k,

            "detection_retention": detection_retention,

            "clean_mean_score": safe_mean(clean_scores),
            "degraded_mean_score_on_clean_refs":
                safe_mean(degraded_scores),

            "clean_mean_iou": safe_mean(clean_ious),
            "degraded_mean_iou_on_clean_refs":
                safe_mean(degraded_ious),
        })

    object_output = Path(args.object_output)
    frame_output = Path(args.frame_output)

    object_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    frame_output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    object_fields = [
        "image_id",
        "ann_id",
        "category_id",
        "condition",
        "severity",
        "clean_detected",
        "degraded_detected",
        "retained_from_clean",
        "clean_score",
        "degraded_score",
        "clean_iou",
        "degraded_iou",
        "score_ratio",
        "iou_ratio",
    ]

    with open(
        object_output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=object_fields,
        )

        writer.writeheader()
        writer.writerows(object_rows)

    frame_fields = [
        "image_id",
        "condition",
        "severity",
        "total_gt",
        "clean_reference_count",
        "degraded_detected_count",
        "retained_count",
        "detection_retention",
        "clean_mean_score",
        "degraded_mean_score_on_clean_refs",
        "clean_mean_iou",
        "degraded_mean_iou_on_clean_refs",
    ]

    with open(
        frame_output,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=frame_fields,
        )

        writer.writeheader()
        writer.writerows(frame_rows)

    print()
    print("Label generation complete")
    print(f"Object labels : {object_output}")
    print(f"Frame summary : {frame_output}")
    print(f"Confidence thr: {args.conf_thr}")
    print(f"IoU thr       : {args.iou_thr}")


if __name__ == "__main__":
    main()