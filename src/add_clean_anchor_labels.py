import argparse
from pathlib import Path

import pandas as pd


def add_clean_anchors(labels: pd.DataFrame) -> pd.DataFrame:
    """
    기존 15개 degraded condition label에
    각 source image의 Clean anchor를 1개씩 추가한다.

    Clean anchor:
        detection_retention = 1.0
        confidence_change = 0.0
        iou_change = 0.0
        confidence_retention_conditional = 1.0
        iou_retention_conditional = 1.0
    """

    required_columns = {
        "image_id",
        "clean_reference_count",
        "k_stratum",
        "split",
    }

    missing = required_columns - set(labels.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # image_id 하나당 Clean anchor 한 개 생성
    # ---------------------------------------------------------
    source_images = (
        labels[
            [
                "image_id",
                "clean_reference_count",
                "k_stratum",
                "split",
            ]
        ]
        .drop_duplicates("image_id")
        .copy()
    )

    source_images["condition"] = "clean"
    source_images["severity"] = "clean"

    # Clean 자기 자신이 기준이므로 모든 reference object 유지
    source_images["retained_count"] = (
        source_images["clean_reference_count"]
    )

    source_images["detection_retention"] = 1.0

    # Clean ↔ Clean 비교이므로 변화량 0
    source_images["confidence_change"] = 0.0
    source_images["iou_change"] = 0.0

    # 유지 비율은 1
    source_images["confidence_retention_conditional"] = 1.0
    source_images["iou_retention_conditional"] = 1.0

    # 기존 CSV와 동일한 column 순서로 맞춤
    for column in labels.columns:
        if column not in source_images.columns:
            source_images[column] = pd.NA

    source_images = source_images[labels.columns]

    # ---------------------------------------------------------
    # Degraded + Clean 합치기
    # ---------------------------------------------------------
    combined = pd.concat(
        [labels, source_images],
        ignore_index=True,
    )

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Add Clean anchor labels to reliability dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input all_frames_v2_split.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV containing degraded + Clean samples",
    )

    args = parser.parse_args()

    labels = pd.read_csv(args.input)

    print("Original rows:", len(labels))
    print("Unique image_ids:", labels["image_id"].nunique())

    combined = add_clean_anchors(labels)

    # ---------------------------------------------------------
    # 검증
    # ---------------------------------------------------------
    clean_rows = combined[
        combined["condition"] == "clean"
    ]

    expected_clean = labels["image_id"].nunique()

    if len(clean_rows) != expected_clean:
        raise RuntimeError(
            f"Expected {expected_clean} Clean anchors, "
            f"but found {len(clean_rows)}."
        )

    # image_id별 Clean sample은 정확히 하나
    clean_count_check = (
        clean_rows
        .groupby("image_id")
        .size()
    )

    if (clean_count_check != 1).any():
        raise RuntimeError(
            "Some image_ids have multiple Clean anchors."
        )

    # 기존 split 유지 확인
    leakage = (
        combined
        .groupby("image_id")["split"]
        .nunique()
    )

    leakage_count = int((leakage > 1).sum())

    if leakage_count != 0:
        raise RuntimeError(
            f"Split leakage detected: {leakage_count}"
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        args.output,
        index=False,
    )

    print("\n=== Clean anchor summary ===")
    print("Clean rows:", len(clean_rows))
    print("Total rows:", len(combined))
    print("Leakage image_ids:", leakage_count)

    print("\nCondition counts:")
    print(combined["condition"].value_counts())

    print("\nSaved:")
    print(args.output)


if __name__ == "__main__":
    main()