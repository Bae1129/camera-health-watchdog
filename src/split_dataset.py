import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def assign_k_stratum(k: int) -> str:
    """Clean reference count K를 stratification group으로 변환."""
    if k == 1:
        return "K1"
    if k == 2:
        return "K2"
    if 3 <= k <= 4:
        return "K3_4"
    if 5 <= k <= 7:
        return "K5_7"
    return "K8_plus"


def split_image_ids(
    image_df: pd.DataFrame,
    seed: int = 42,
) -> dict:
    """
    image_id 단위로 stratified group split 수행.

    각 K stratum 내부에서
    Train / Validation / Test = 70 / 15 / 15
    """

    rng = np.random.default_rng(seed)

    split_map = {}

    for stratum, group in image_df.groupby("k_stratum"):

        image_ids = group["image_id"].to_numpy().copy()
        rng.shuffle(image_ids)

        n = len(image_ids)

        n_train = int(np.floor(n * 0.70))
        n_val = int(np.floor(n * 0.15))

        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train:n_train + n_val]
        test_ids = image_ids[n_train + n_val:]

        for image_id in train_ids:
            split_map[image_id] = "train"

        for image_id in val_ids:
            split_map[image_id] = "val"

        for image_id in test_ids:
            split_map[image_id] = "test"

        print(
            f"{stratum:8s} | total={n:4d} "
            f"| train={len(train_ids):4d} "
            f"| val={len(val_ids):4d} "
            f"| test={len(test_ids):4d}"
        )

    return split_map


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Split reliability labels by image_id while stratifying "
            "on Clean reference count K."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input all_frames_v2.csv",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV with split labels",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    labels = pd.read_csv(args.input)

    required_columns = {
        "image_id",
        "clean_reference_count",
    }

    missing = required_columns - set(labels.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # ---------------------------------------------------------
    # image_id마다 Clean reference count K가 하나인지 검증
    # ---------------------------------------------------------
    k_check = (
        labels
        .groupby("image_id")["clean_reference_count"]
        .nunique()
    )

    if (k_check != 1).any():
        raise ValueError(
            "Some image_ids have multiple clean_reference_count values."
        )

    # ---------------------------------------------------------
    # 원본 image_id당 한 행
    # ---------------------------------------------------------
    image_df = (
        labels[
            ["image_id", "clean_reference_count"]
        ]
        .drop_duplicates("image_id")
        .copy()
    )

    image_df["k_stratum"] = (
        image_df["clean_reference_count"]
        .astype(int)
        .map(assign_k_stratum)
    )

    print("\n=== K stratification distribution ===")

    print(
        image_df["k_stratum"]
        .value_counts()
        .reindex(
            ["K1", "K2", "K3_4", "K5_7", "K8_plus"]
        )
    )

    print("\n=== Split ===")

    split_map = split_image_ids(
        image_df,
        seed=args.seed,
    )

    # ---------------------------------------------------------
    # 15개 파생 샘플에 동일 split 부여
    # ---------------------------------------------------------
    labels["k_stratum"] = (
        labels["clean_reference_count"]
        .astype(int)
        .map(assign_k_stratum)
    )

    labels["split"] = labels["image_id"].map(split_map)

    if labels["split"].isna().any():
        raise RuntimeError(
            "Some rows were not assigned to a split."
        )

    # ---------------------------------------------------------
    # Leakage 검증
    # ---------------------------------------------------------
    leakage = (
        labels
        .groupby("image_id")["split"]
        .nunique()
    )

    leakage_count = int((leakage > 1).sum())

    if leakage_count != 0:
        raise RuntimeError(
            f"Data leakage detected: {leakage_count} image_ids"
        )

    # ---------------------------------------------------------
    # 저장
    # ---------------------------------------------------------
    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labels.to_csv(
        args.output,
        index=False,
    )

    print("\n=== Final split summary ===")

    summary = (
        labels[
            ["image_id", "k_stratum", "split"]
        ]
        .drop_duplicates("image_id")
        .groupby(["k_stratum", "split"])
        .size()
        .unstack(fill_value=0)
    )

    print(summary)

    print("\nUnique source images:")
    print(
        labels
        .groupby("split")["image_id"]
        .nunique()
    )

    print("\nDerived samples:")
    print(labels["split"].value_counts())

    print("\nLeakage image_ids:", leakage_count)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()