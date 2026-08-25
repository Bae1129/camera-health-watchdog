import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_frame_labels(object_csv: Path) -> pd.DataFrame:
    obj = pd.read_csv(object_csv)

    rows = []

    group_cols = ["image_id", "condition", "severity"]

    for (image_id, condition, severity), g in obj.groupby(group_cols):

        # Clean에서 검출 가능했던 객체
        clean_ref = g[g["clean_detected"] == 1]
        K = len(clean_ref)

        # Detection retention 정의 불가능
        if K == 0:
            continue

        # Clean과 degraded 모두에서 검출된 동일 객체
        retained = clean_ref[
            clean_ref["retained_from_clean"] == 1
        ]

        k = len(retained)

        # 1. Detection capability retention
        detection_retention = k / K

        # 2. 살아남은 객체의 detection quality 변화
        # 미검출 객체를 0으로 넣지 않음
        if k > 0:
            confidence_change = (
                retained["degraded_score"]
                - retained["clean_score"]
            ).mean()

            iou_change = (
                retained["degraded_iou"]
                - retained["clean_iou"]
            ).mean()

            confidence_retention_conditional = (
                retained["degraded_score"]
                / retained["clean_score"]
            ).mean()

            iou_retention_conditional = (
                retained["degraded_iou"]
                / retained["clean_iou"]
            ).mean()

        else:
            # 비교 가능한 prediction 자체가 없음
            confidence_change = np.nan
            iou_change = np.nan
            confidence_retention_conditional = np.nan
            iou_retention_conditional = np.nan

        rows.append({
            "image_id": image_id,
            "condition": condition,
            "severity": severity,

            "clean_reference_count": K,
            "retained_count": k,

            "detection_retention": detection_retention,

            "confidence_change": confidence_change,
            "iou_change": iou_change,

            "confidence_retention_conditional":
                confidence_retention_conditional,

            "iou_retention_conditional":
                iou_retention_conditional,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--object-csv",
        required=True,
        type=Path,
        help="Object-level reliability label CSV",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output frame-level CSV",
    )

    args = parser.parse_args()

    frame_df = build_frame_labels(args.object_csv)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_df.to_csv(
        args.output,
        index=False,
    )

    print(f"Frame rows: {len(frame_df)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()