from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CONDITION_PATH_MAP = {
    ("defocus", "d3"): "defocus/S1_d3",
    ("defocus", "d7"): "defocus/S2_d7",
    ("defocus", "d11"): "defocus/S3_d11",
    ("defocus", "d15"): "defocus/S4_d15",
    ("defocus", "d21"): "defocus/S5_d21",

    ("motion_blur", "d3"): "motion_blur/M1_d3",
    ("motion_blur", "d7"): "motion_blur/M2_d7",
    ("motion_blur", "d11"): "motion_blur/M3_d11",
    ("motion_blur", "d15"): "motion_blur/M4_d15",
    ("motion_blur", "d21"): "motion_blur/M5_d21",

    ("gaussian_noise", "sigma5"): "gaussian_noise/N1_sigma5",
    ("gaussian_noise", "sigma10"): "gaussian_noise/N2_sigma10",
    ("gaussian_noise", "sigma15"): "gaussian_noise/N3_sigma15",
    ("gaussian_noise", "sigma20"): "gaussian_noise/N4_sigma20",
    ("gaussian_noise", "sigma25"): "gaussian_noise/N5_sigma25",
}


class ReliabilityDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split,
        clean_root,
        degradation_root,
        transform=None,
    ):
        self.csv_path = Path(csv_path)
        self.clean_root = Path(clean_root)
        self.degradation_root = Path(degradation_root)
        self.transform = transform

        df = pd.read_csv(self.csv_path)

        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.df) == 0:
            raise ValueError(
                f"No samples found for split='{split}'"
            )

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, row):
        image_id = int(row["image_id"])
        filename = f"{image_id:012d}.jpg"

        condition = row["condition"]
        severity = row["severity"]

        # Clean image
        if condition == "clean":
            return self.clean_root / filename

        # Degraded image
        key = (condition, severity)

        if key not in CONDITION_PATH_MAP:
            raise ValueError(
                f"Unknown condition/severity: {key}"
            )

        relative_dir = CONDITION_PATH_MAP[key]

        return (
            self.degradation_root
            / relative_dir
            / filename
        )

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # -----------------------------------------------------
        # 1. Image
        # -----------------------------------------------------
        image_path = self._get_image_path(row)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # -----------------------------------------------------
        # 2. Detection Retention
        # 항상 정의 가능
        # -----------------------------------------------------
        detection_retention = float(
            row["detection_retention"]
        )

        # -----------------------------------------------------
        # 3. Confidence / IoU
        #
        # retained_count > 0:
        #   실제 비교 가능한 surviving object 존재
        #
        # retained_count == 0:
        #   Confidence / IoU 변화는 정의 불가
        # -----------------------------------------------------
        retained_count = int(row["retained_count"])

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
            # -------------------------------------------------
            # 이 0은 실제 정답이 아니다.
            #
            # NaN을 tensor에 넣지 않기 위한 placeholder이며,
            # train loss 계산에서 quality_valid_mask=0으로
            # 완전히 제외해야 한다.
            # -------------------------------------------------
            confidence_change = 0.0
            iou_change = 0.0

            quality_valid_mask = 0.0

        # -----------------------------------------------------
        # 4. Multi-task targets
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

            # [R_det, Delta_conf, Delta_iou]
            "targets": targets,

            # Confidence / IoU loss 사용 여부
            "quality_valid_mask": quality_valid_mask,

            # 분석 / debugging용
            "image_id": int(row["image_id"]),
            "condition": row["condition"],
            "severity": row["severity"],
            "clean_reference_count": int(
                row["clean_reference_count"]
            ),
            "retained_count": retained_count,
        }