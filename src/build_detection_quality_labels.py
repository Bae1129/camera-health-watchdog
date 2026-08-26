import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EPS = 1e-12


# ============================================================
# IoU
# ============================================================

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


# ============================================================
# Load COCO GT
# ============================================================

def load_coco_gt(annotation_path):
    """
    Load COCO GT annotations.

    Crowd annotations are excluded,
    consistent with the previous labeling pipeline.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    gt_by_image = defaultdict(list)

    for ann in coco["annotations"]:

        if ann.get("iscrowd", 0) == 1:
            continue

        gt_by_image[ann["image_id"]].append({
            "ann_id": int(ann["id"]),
            "category_id": int(ann["category_id"]),
            "bbox": ann["bbox"],
        })

    return gt_by_image


# ============================================================
# Load YOLO predictions
# ============================================================

def load_predictions(prediction_path):
    """
    Load detector predictions in COCO result format.
    """
    with open(prediction_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)

    for pred in predictions:

        pred_by_image[int(pred["image_id"])].append({
            "image_id": int(pred["image_id"]),
            "category_id": int(pred["category_id"]),
            "bbox": pred["bbox"],
            "score": float(pred["score"]),
        })

    return pred_by_image


# ============================================================
# Prediction <-> GT association
# ============================================================

def match_predictions_to_gt(
    gt_list,
    pred_list,
    conf_thr=0.25,
):
    """
    Same-class, one-to-one prediction-to-GT association.

    Important
    ---------
    Confidence threshold:
        Used as the minimum operating confidence for a valid
        detector output.

    IoU threshold:
        NOT used.

    IoU is retained as a continuous localization-quality value.

    Matching itself is performed greedily from highest IoU pair
    to lowest IoU pair.

    IoU == 0 pairs are ignored because there is no spatial overlap.
    This is not a performance threshold such as IoU >= 0.5.
    """

    valid_predictions = [
        pred
        for pred in pred_list
        if pred["score"] >= conf_thr
    ]

    candidates = []

    # --------------------------------------------------------
    # Build all same-class GT / prediction candidate pairs
    # --------------------------------------------------------
    for gt_idx, gt in enumerate(gt_list):

        for pred_idx, pred in enumerate(valid_predictions):

            if gt["category_id"] != pred["category_id"]:
                continue

            iou = bbox_iou_xywh(
                gt["bbox"],
                pred["bbox"],
            )

            # No spatial overlap -> no meaningful association.
            if iou <= 0.0:
                continue

            candidates.append({
                "gt_idx": gt_idx,
                "pred_idx": pred_idx,
                "iou": float(iou),
            })

    # Highest-IoU pair first
    candidates.sort(
        key=lambda x: x["iou"],
        reverse=True,
    )

    used_gt = set()
    used_pred = set()

    matches = {}

    # --------------------------------------------------------
    # Greedy one-to-one association
    # --------------------------------------------------------
    for candidate in candidates:

        gt_idx = candidate["gt_idx"]
        pred_idx = candidate["pred_idx"]

        if gt_idx in used_gt:
            continue

        if pred_idx in used_pred:
            continue

        gt = gt_list[gt_idx]
        pred = valid_predictions[pred_idx]

        ann_id = gt["ann_id"]

        matches[ann_id] = {
            "score": float(pred["score"]),
            "iou": float(candidate["iou"]),
            "bbox": pred["bbox"],
        }

        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

    return matches


# ============================================================
# Detection quality
# ============================================================

def get_detection_quality(match):
    """
    Object-level continuous detection quality.

        Q = confidence * IoU

    Unmatched object:
        Q = 0
    """
    if match is None:
        return 0.0

    return float(
        match["score"] * match["iou"]
    )


# ============================================================
# Label generation
# ============================================================

def build_labels(
    gt_by_image,
    clean_pred_by_image,
    degraded_pred_by_image,
    condition,
    severity,
    conf_thr,
):
    object_rows = []
    frame_rows = []

    skipped_no_clean_quality = 0

    for image_id, gt_list in gt_by_image.items():

        clean_predictions = clean_pred_by_image.get(
            image_id,
            [],
        )

        degraded_predictions = degraded_pred_by_image.get(
            image_id,
            [],
        )

        # ====================================================
        # Clean / degraded use EXACTLY the same association rule
        # ====================================================

        clean_matches = match_predictions_to_gt(
            gt_list,
            clean_predictions,
            conf_thr=conf_thr,
        )

        degraded_matches = match_predictions_to_gt(
            gt_list,
            degraded_predictions,
            conf_thr=conf_thr,
        )

        clean_quality_sum = 0.0
        degraded_quality_sum = 0.0
        retained_quality_sum = 0.0

        clean_matched_count = 0
        degraded_matched_count = 0

        image_object_rows = []

        # ====================================================
        # Object-level quality
        # ====================================================

        for gt in gt_list:

            ann_id = gt["ann_id"]

            clean_match = clean_matches.get(
                ann_id
            )

            degraded_match = degraded_matches.get(
                ann_id
            )

            # -----------------------------------------------
            # Clean
            # -----------------------------------------------

            if clean_match is not None:

                clean_score = float(
                    clean_match["score"]
                )

                clean_iou = float(
                    clean_match["iou"]
                )

                clean_matched_count += 1

            else:

                clean_score = 0.0
                clean_iou = 0.0

            clean_quality = get_detection_quality(
                clean_match
            )

            # -----------------------------------------------
            # Degraded
            # -----------------------------------------------

            if degraded_match is not None:

                degraded_score = float(
                    degraded_match["score"]
                )

                degraded_iou = float(
                    degraded_match["iou"]
                )

                degraded_matched_count += 1

            else:

                degraded_score = 0.0
                degraded_iou = 0.0

            degraded_quality = get_detection_quality(
                degraded_match
            )

            # -----------------------------------------------
            # Retained quality
            #
            # Improvement beyond Clean does not compensate
            # for losses of other objects.
            # -----------------------------------------------

            retained_quality = min(
                clean_quality,
                degraded_quality,
            )

            if clean_quality > EPS:

                object_quality_retention = (
                    retained_quality
                    / clean_quality
                )

            else:

                # No Clean baseline performance for this GT.
                object_quality_retention = None

            clean_quality_sum += clean_quality
            degraded_quality_sum += degraded_quality
            retained_quality_sum += retained_quality

            image_object_rows.append({
                "image_id": image_id,
                "ann_id": ann_id,
                "category_id": gt["category_id"],

                "condition": condition,
                "severity": severity,

                "clean_score": clean_score,
                "clean_iou": clean_iou,
                "clean_quality": clean_quality,

                "degraded_score": degraded_score,
                "degraded_iou": degraded_iou,
                "degraded_quality": degraded_quality,

                "retained_quality": retained_quality,

                "object_quality_retention":
                    object_quality_retention,
            })

        # ====================================================
        # No Clean detector performance -> retention undefined
        # ====================================================

        if clean_quality_sum <= EPS:

            skipped_no_clean_quality += 1
            continue

        # ====================================================
        # Frame-level Detection Quality Retention
        #
        #          sum min(Q_clean, Q_current)
        # R_Q = --------------------------------
        #                sum Q_clean
        #
        # Range: [0, 1]
        # ====================================================

        detection_quality_retention = (
            retained_quality_sum
            / clean_quality_sum
        )

        # Numerical safety only
        detection_quality_retention = max(
            0.0,
            min(
                1.0,
                detection_quality_retention,
            ),
        )

        # Only keep object rows belonging to valid frames
        object_rows.extend(
            image_object_rows
        )

        frame_rows.append({
            "image_id": image_id,

            "condition": condition,
            "severity": severity,

            "total_gt": len(gt_list),

            "clean_matched_count":
                clean_matched_count,

            "degraded_matched_count":
                degraded_matched_count,

            "clean_quality_sum":
                clean_quality_sum,

            "degraded_quality_sum":
                degraded_quality_sum,

            "retained_quality_sum":
                retained_quality_sum,

            "detection_quality_retention":
                detection_quality_retention,
        })

    return (
        object_rows,
        frame_rows,
        skipped_no_clean_quality,
    )


# ============================================================
# CSV save
# ============================================================

def save_object_csv(rows, output_path):

    fields = [
        "image_id",
        "ann_id",
        "category_id",

        "condition",
        "severity",

        "clean_score",
        "clean_iou",
        "clean_quality",

        "degraded_score",
        "degraded_iou",
        "degraded_quality",

        "retained_quality",
        "object_quality_retention",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def save_frame_csv(rows, output_path):

    fields = [
        "image_id",

        "condition",
        "severity",

        "total_gt",

        "clean_matched_count",
        "degraded_matched_count",

        "clean_quality_sum",
        "degraded_quality_sum",
        "retained_quality_sum",

        "detection_quality_retention",
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build continuous Detection Quality Retention labels "
            "from COCO GT and YOLO predictions."
        )
    )

    parser.add_argument(
        "--ann",
        required=True,
        type=Path,
        help="COCO annotation JSON",
    )

    parser.add_argument(
        "--clean-pred",
        required=True,
        type=Path,
        help="Clean YOLO prediction JSON",
    )

    parser.add_argument(
        "--degraded-pred",
        type=Path,
        default=None,
        help=(
            "Current/degraded YOLO prediction JSON. "
            "If omitted, Clean is compared with itself."
        ),
    )

    parser.add_argument(
        "--condition",
        type=str,
        default="clean",
    )

    parser.add_argument(
        "--severity",
        type=str,
        default="clean",
    )

    parser.add_argument(
        "--conf-thr",
        type=float,
        default=0.25,
        help=(
            "Minimum confidence for a prediction "
            "to be considered a valid detector output."
        ),
    )

    parser.add_argument(
        "--object-output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--frame-output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    # ========================================================
    # Clean mode
    # ========================================================

    if args.degraded_pred is None:

        degraded_pred_path = args.clean_pred

        condition = "clean"
        severity = "clean"

    else:

        degraded_pred_path = args.degraded_pred

        condition = args.condition
        severity = args.severity

    # ========================================================
    # Load
    # ========================================================

    gt_by_image = load_coco_gt(
        args.ann
    )

    clean_pred_by_image = load_predictions(
        args.clean_pred
    )

    degraded_pred_by_image = load_predictions(
        degraded_pred_path
    )

    # ========================================================
    # Build
    # ========================================================

    (
        object_rows,
        frame_rows,
        skipped_count,
    ) = build_labels(
        gt_by_image=gt_by_image,
        clean_pred_by_image=clean_pred_by_image,
        degraded_pred_by_image=degraded_pred_by_image,
        condition=condition,
        severity=severity,
        conf_thr=args.conf_thr,
    )

    # ========================================================
    # Save
    # ========================================================

    save_object_csv(
        object_rows,
        args.object_output,
    )

    save_frame_csv(
        frame_rows,
        args.frame_output,
    )

    # ========================================================
    # Summary
    # ========================================================

    retention_values = [
        row["detection_quality_retention"]
        for row in frame_rows
    ]

    print()
    print("==============================================")
    print("Detection Quality Label Generation Complete")
    print("==============================================")

    print(f"Condition        : {condition}")
    print(f"Severity         : {severity}")
    print(f"Confidence thr   : {args.conf_thr}")

    print()
    print(f"Valid frames     : {len(frame_rows)}")
    print(f"Object rows      : {len(object_rows)}")
    print(
        f"Skipped frames   : "
        f"{skipped_count} "
        f"(clean quality = 0)"
    )

    if retention_values:

        mean_retention = (
            sum(retention_values)
            / len(retention_values)
        )

        print()
        print(
            "Mean quality retention : "
            f"{mean_retention:.6f}"
        )

        print(
            "Min quality retention  : "
            f"{min(retention_values):.6f}"
        )

        print(
            "Max quality retention  : "
            f"{max(retention_values):.6f}"
        )

    print()
    print(
        f"Object CSV : "
        f"{args.object_output}"
    )

    print(
        f"Frame CSV  : "
        f"{args.frame_output}"
    )


if __name__ == "__main__":
    main()