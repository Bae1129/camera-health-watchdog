from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.degradations import (
    MOTION_ANGLES,
    apply_defocus,
    apply_gaussian_noise,
    apply_motion_blur,
)


class RandomReliabilityDataset(Dataset):
    """
    On-the-fly reliability dataset.

    Train:
        - source image 1장당 매 epoch 하나의 condition만 선택
        - 따라서 len(dataset) = source image 개수

    Val/Test:
        - CSV에 존재하는 모든 condition을 고정적으로 사용
        - 따라서 모든 16-state를 평가

    Label:
        - 기존 all_frames_final.csv 그대로 사용
        - Rdet, DeltaConf, DeltaIoU
        - K = clean_reference_count
        - k = retained_count
    """

    def __init__(
        self,
        csv_path,
        split,
        clean_root,
        transform=None,
        seed=42,
        random_per_source=None,
    ):
        self.csv_path = Path(csv_path)
        self.clean_root = Path(clean_root)
        self.transform = transform
        self.seed = int(seed)

        # 현재 epoch
        self.epoch = 0

        df = pd.read_csv(self.csv_path)

        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split='{split}'"
            )

        # 기본 동작:
        # train → source당 랜덤 1개
        # val/test → 모든 상태 고정 평가
        if random_per_source is None:
            random_per_source = split == "train"

        self.random_per_source = random_per_source

        # -----------------------------------------------------
        # Train용:
        # image_id별로 CSV row index를 묶어둔다.
        # -----------------------------------------------------
        if self.random_per_source:
            self.image_ids = sorted(
                self.df["image_id"]
                .astype(int)
                .unique()
                .tolist()
            )

            self.rows_by_image = {}

            for image_id, group in self.df.groupby("image_id"):
                self.rows_by_image[int(image_id)] = (
                    group.index.tolist()
                )

        print(
            f"[RandomReliabilityDataset] "
            f"split={split}, "
            f"rows={len(self.df)}, "
            f"random_per_source={self.random_per_source}"
        )

        if self.random_per_source:
            print(
                f"Unique source images: "
                f"{len(self.image_ids)}"
            )

    # =========================================================
    # Epoch control
    # =========================================================

    def set_epoch(self, epoch):
        """
        train loop에서 epoch마다 호출.

        같은 seed라면 동일 실험을 다시 수행했을 때
        같은 random sampling sequence를 재현할 수 있다.
        """
        self.epoch = int(epoch)

    # =========================================================
    # Dataset length
    # =========================================================

    def __len__(self):

        if self.random_per_source:
            # source당 1개
            return len(self.image_ids)

        # val/test는 모든 condition
        return len(self.df)

    # =========================================================
    # Random row selection
    # =========================================================

    def _select_row(self, index):

        # -----------------------------------------------------
        # Val/Test
        # → CSV row 그대로
        # -----------------------------------------------------
        if not self.random_per_source:
            return self.df.iloc[index]

        # -----------------------------------------------------
        # Train
        # → source image 하나 선택
        # → 해당 source의 16 state 중 하나 선택
        # -----------------------------------------------------
        image_id = self.image_ids[index]

        candidate_indices = (
            self.rows_by_image[image_id]
        )

        # source + epoch 기반 deterministic random
        #
        # epoch이 달라지면 다른 state가 선택되지만
        # 같은 seed로 다시 실행하면 재현 가능
        rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + image_id
        )

        selected_index = rng.choice(
            candidate_indices
        )

        return self.df.iloc[selected_index]

    # =========================================================
    # Motion angle
    # =========================================================

    @staticmethod
    def _motion_angle(image_id):
        """
        Source image별 motion direction 고정.

        동일 source는 severity가 달라도 같은 angle 사용.
        """
        return MOTION_ANGLES[
            image_id % len(MOTION_ANGLES)
        ]

    # =========================================================
    # JPEG round-trip
    # =========================================================

    @staticmethod
    def _jpeg_roundtrip(image):
        """
        기존 degraded dataset 생성 시
        JPEG quality=100으로 저장했던 과정을 메모리에서 재현.

        Disk에는 저장하지 않는다.
        """

        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )

        if not success:
            raise RuntimeError(
                "JPEG encoding failed."
            )

        decoded = cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )

        return decoded

    # =========================================================
    # On-the-fly degradation
    # =========================================================

    def _apply_condition(
        self,
        clean_image,
        row,
    ):
        image_id = int(row["image_id"])
        condition = str(row["condition"])
        severity = str(row["severity"])

        # -----------------------------------------------------
        # Clean
        # -----------------------------------------------------
        if condition == "clean":
            return clean_image.copy()

        # -----------------------------------------------------
        # Defocus
        #
        # severity:
        # d3 / d7 / d11 / d15 / d21
        # -----------------------------------------------------
        if condition == "defocus":

            diameter = int(
                severity.replace("d", "")
            )

            degraded = apply_defocus(
                clean_image,
                diameter=diameter,
            )

        # -----------------------------------------------------
        # Motion Blur
        # -----------------------------------------------------
        elif condition == "motion_blur":

            length = int(
                severity.replace("d", "")
            )

            angle = self._motion_angle(
                image_id
            )

            degraded = apply_motion_blur(
                clean_image,
                length=length,
                angle=angle,
            )

        # -----------------------------------------------------
        # Gaussian Noise
        #
        # severity:
        # sigma5 / sigma10 / ...
        # -----------------------------------------------------
        elif condition == "gaussian_noise":

            sigma = float(
                severity.replace("sigma", "")
            )

            # 기존 규칙:
            # source마다 동일 noise pattern,
            # severity에 따라 sigma만 증가
            noise_seed = 42 + image_id

            degraded = apply_gaussian_noise(
                clean_image,
                sigma=sigma,
                seed=noise_seed,
            )

        else:
            raise ValueError(
                f"Unknown condition: {condition}"
            )

        # 기존 synthetic dataset이
        # JPEG quality=100으로 저장된 후 사용되었으므로
        # 같은 압축 과정을 메모리에서 재현
        degraded = self._jpeg_roundtrip(
            degraded
        )

        return degraded

    # =========================================================
    # Get item
    # =========================================================

    def __getitem__(self, index):

        row = self._select_row(index)

        image_id = int(row["image_id"])

        filename = f"{image_id:012d}.jpg"

        clean_path = (
            self.clean_root / filename
        )

        if not clean_path.exists():
            raise FileNotFoundError(
                f"Clean image not found: "
                f"{clean_path}"
            )

        # -----------------------------------------------------
        # 1. Clean image load
        # -----------------------------------------------------
        clean_image = cv2.imread(
            str(clean_path)
        )

        if clean_image is None:
            raise FileNotFoundError(
                f"Could not read: {clean_path}"
            )

        # -----------------------------------------------------
        # 2. On-the-fly degradation
        # -----------------------------------------------------
        image = self._apply_condition(
            clean_image,
            row,
        )

        # OpenCV BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        # =====================================================
        # Labels
        # =====================================================

        # -----------------------------------------------------
        # Detection Retention
        # -----------------------------------------------------
        detection_retention = float(
            row["detection_retention"]
        )

        # K
        clean_reference_count = int(
            row["clean_reference_count"]
        )

        # k
        retained_count = int(
            row["retained_count"]
        )

        # -----------------------------------------------------
        # Confidence / IoU
        # -----------------------------------------------------
        quality_valid = retained_count > 0

        if quality_valid:

            confidence_change = float(
                row["confidence_change"]
            )

            iou_change = float(
                row["iou_change"]
            )

            quality_valid_mask = 1.0

        else:
            # 실제 label 0이 아니라
            # tensor용 placeholder
            confidence_change = 0.0
            iou_change = 0.0

            quality_valid_mask = 0.0

        # -----------------------------------------------------
        # Multi-task target
        # -----------------------------------------------------
        targets = torch.tensor(
            [
                detection_retention,
                confidence_change,
                iou_change,
            ],
            dtype=torch.float32,
        )

        quality_valid_mask = torch.tensor(
            quality_valid_mask,
            dtype=torch.float32,
        )

        return {
            "image": image,

            # [Rdet, DeltaConf, DeltaIoU]
            "targets": targets,

            # Auxiliary loss mask
            "quality_valid_mask":
                quality_valid_mask,

            # Binomial loss에서 사용
            "clean_reference_count":
                clean_reference_count,

            "retained_count":
                retained_count,

            # debugging / analysis
            "image_id": image_id,
            "condition": str(
                row["condition"]
            ),
            "severity": str(
                row["severity"]
            ),
        }