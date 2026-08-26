

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from src.model_multitask import MultiTaskReliabilityModel


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Folder mapping
# CSV severity -> pre-generated folder
# ============================================================

DEFOCUS_FOLDER = {
    "d3": "S1_d3",
    "d7": "S2_d7",
    "d11": "S3_d11",
    "d15": "S4_d15",
    "d21": "S5_d21",
}

MOTION_FOLDER = {
    "d3": "M1_d3",
    "d7": "M2_d7",
    "d11": "M3_d11",
    "d15": "M4_d15",
    "d21": "M5_d21",
}

NOISE_FOLDER = {
    "sigma5": "N1_sigma5",
    "sigma10": "N2_sigma10",
    "sigma15": "N3_sigma15",
    "sigma20": "N4_sigma20",
    "sigma25": "N5_sigma25",
}


# ============================================================
# Dataset
# ============================================================

class RandomFileReliabilityDataset(Dataset):

    def __init__(
        self,
        csv_path,
        split,
        clean_root,
        degradation_root,
        transform=None,
        seed=42,
        random_per_source=None,
    ):
        self.clean_root = Path(clean_root)
        self.degradation_root = Path(degradation_root)

        self.transform = transform
        self.seed = int(seed)
        self.epoch = 0

        df = pd.read_csv(csv_path)

        self.df = (
            df[df["split"] == split]
            .reset_index(drop=True)
        )

        if len(self.df) == 0:
            raise ValueError(
                f"No data for split={split}"
            )

        if random_per_source is None:
            random_per_source = (
                split == "train"
            )

        self.random_per_source = (
            random_per_source
        )

        # --------------------------------------------
        # Train:
        # image_id별 16 states 묶기
        # --------------------------------------------
        if self.random_per_source:

            self.image_ids = sorted(
                self.df["image_id"]
                .astype(int)
                .unique()
                .tolist()
            )

            self.rows_by_image = {}

            for image_id, group in (
                self.df.groupby("image_id")
            ):
                self.rows_by_image[
                    int(image_id)
                ] = group.index.tolist()

            counts = [
                len(v)
                for v in
                self.rows_by_image.values()
            ]

            print(
                f"[TRAIN] sources="
                f"{len(self.image_ids)}"
            )

            print(
                "States/source:",
                min(counts),
                "~",
                max(counts),
            )

        else:
            print(
                f"[{split.upper()}] "
                f"samples={len(self.df)}"
            )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):

        if self.random_per_source:
            return len(self.image_ids)

        return len(self.df)

    def _select_row(self, index):

        # --------------------------------------------
        # Val/Test:
        # 모든 row 고정 사용
        # --------------------------------------------
        if not self.random_per_source:
            return self.df.iloc[index]

        # --------------------------------------------
        # Train:
        # source 하나당 16 state 중 하나 랜덤
        # --------------------------------------------
        image_id = self.image_ids[index]

        candidates = (
            self.rows_by_image[image_id]
        )

        # epoch + source 기반 deterministic random
        rng = np.random.default_rng(
            self.seed
            + self.epoch * 1_000_003
            + image_id
        )

        selected = rng.choice(candidates)

        return self.df.iloc[selected]

    def _get_image_path(self, row):

        image_id = int(row["image_id"])
        condition = str(row["condition"])
        severity = str(row["severity"])

        filename = (
            f"{image_id:012d}.jpg"
        )

        # --------------------------------------------
        # Clean
        # --------------------------------------------
        if condition == "clean":

            return (
                self.clean_root
                / filename
            )

        # --------------------------------------------
        # Defocus
        # --------------------------------------------
        if condition == "defocus":

            folder = (
                DEFOCUS_FOLDER[severity]
            )

            return (
                self.degradation_root
                / "defocus"
                / folder
                / filename
            )

        # --------------------------------------------
        # Motion Blur
        # --------------------------------------------
        if condition == "motion_blur":

            folder = (
                MOTION_FOLDER[severity]
            )

            return (
                self.degradation_root
                / "motion_blur"
                / folder
                / filename
            )

        # --------------------------------------------
        # Gaussian Noise
        # --------------------------------------------
        if condition == "gaussian_noise":

            folder = (
                NOISE_FOLDER[severity]
            )

            return (
                self.degradation_root
                / "gaussian_noise"
                / folder
                / filename
            )

        raise ValueError(
            f"Unknown condition: "
            f"{condition}"
        )

    def __getitem__(self, index):

        row = self._select_row(index)

        image_path = (
            self._get_image_path(row)
        )

        if not image_path.exists():
            raise FileNotFoundError(
                image_path
            )

        image = (
            Image.open(image_path)
            .convert("RGB")
        )

        if self.transform:
            image = self.transform(image)

        K = int(
            row["clean_reference_count"]
        )

        k = int(
            row["retained_count"]
        )

        rdet = float(
            row["detection_retention"]
        )

        # k=0이면 auxiliary quality undefined
        if k > 0:

            conf = float(
                row["confidence_change"]
            )

            iou = float(
                row["iou_change"]
            )

            mask = 1.0

        else:

            conf = 0.0
            iou = 0.0
            mask = 0.0

        targets = torch.tensor(
            [
                rdet,
                conf,
                iou,
            ],
            dtype=torch.float32,
        )

        return {
            "image": image,
            "targets": targets,

            "quality_valid_mask":
                torch.tensor(
                    mask,
                    dtype=torch.float32,
                ),

            "clean_reference_count": K,
            "retained_count": k,

            "image_id":
                int(row["image_id"]),

            "condition":
                str(row["condition"]),

            "severity":
                str(row["severity"]),
        }


# ============================================================
# Transform
# ============================================================

def build_transform():

    return transforms.Compose([
        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ])


# ============================================================
# Loss
# ============================================================

def calculate_loss(
    outputs,
    targets,
    mask,
    K,
    k,
    det_loss_type,
):

    pred_det = outputs[:, 0]
    pred_conf = outputs[:, 1]
    pred_iou = outputs[:, 2]

    target_det = targets[:, 0]
    target_conf = targets[:, 1]
    target_iou = targets[:, 2]

    # --------------------------------------------
    # Detection loss
    # --------------------------------------------
    if det_loss_type == "mae":

        det_loss = F.l1_loss(
            pred_det,
            target_det,
        )

    elif det_loss_type == "binomial":

        K = K.float()
        k = k.float()

        eps = 1e-6

        p = pred_det.clamp(
            eps,
            1.0 - eps,
        )

        nll = -(
            k * torch.log(p)
            +
            (K - k)
            * torch.log(1.0 - p)
        )

        # object observation 기준 normalize
        # K 큰 frame은 더 큰 영향 유지
        det_loss = (
            nll.sum()
            /
            K.sum().clamp(min=1.0)
        )

    else:

        raise ValueError(
            det_loss_type
        )

    # --------------------------------------------
    # Auxiliary masked MAE
    # --------------------------------------------
    mask = mask.float()

    valid_count = (
        mask.sum()
        .clamp(min=1.0)
    )

    conf_loss = (
        torch.abs(
            pred_conf - target_conf
        )
        * mask
    ).sum() / valid_count

    iou_loss = (
        torch.abs(
            pred_iou - target_iou
        )
        * mask
    ).sum() / valid_count

    total_loss = (
        det_loss
        + conf_loss
        + iou_loss
    )

    return (
        total_loss,
        det_loss,
        conf_loss,
        iou_loss,
    )


# ============================================================
# Train
# ============================================================

def train_epoch(
    model,
    loader,
    optimizer,
    device,
    det_loss_type,
):

    model.train()

    sums = np.zeros(4)
    n_batches = 0

    for batch in tqdm(
        loader,
        desc="Train",
        leave=False,
    ):

        images = (
            batch["image"]
            .to(device)
        )

        targets = (
            batch["targets"]
            .to(device)
        )

        mask = (
            batch[
                "quality_valid_mask"
            ]
            .to(device)
        )

        K = (
            batch[
                "clean_reference_count"
            ]
            .to(device)
        )

        k = (
            batch[
                "retained_count"
            ]
            .to(device)
        )

        optimizer.zero_grad()

        outputs = model(images)

        losses = calculate_loss(
            outputs,
            targets,
            mask,
            K,
            k,
            det_loss_type,
        )

        losses[0].backward()
        optimizer.step()

        sums += np.array([
            x.item()
            for x in losses
        ])

        n_batches += 1

    return (
        sums
        / max(n_batches, 1)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    det_sum = 0.0
    det_n = 0

    conf_sum = 0.0
    conf_n = 0

    iou_sum = 0.0
    iou_n = 0

    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):

        images = (
            batch["image"]
            .to(device)
        )

        targets = (
            batch["targets"]
            .to(device)
        )

        mask = (
            batch[
                "quality_valid_mask"
            ]
            .to(device)
            .bool()
        )

        outputs = model(images)

        det_sum += torch.abs(
            outputs[:, 0]
            - targets[:, 0]
        ).sum().item()

        det_n += len(images)

        if mask.any():

            conf_sum += torch.abs(
                outputs[mask, 1]
                - targets[mask, 1]
            ).sum().item()

            conf_n += (
                mask.sum().item()
            )

            iou_sum += torch.abs(
                outputs[mask, 2]
                - targets[mask, 2]
            ).sum().item()

            iou_n += (
                mask.sum().item()
            )

    return {
        "val_det_mae":
            det_sum / det_n,

        "val_conf_mae":
            conf_sum
            / max(conf_n, 1),

        "val_iou_mae":
            iou_sum
            / max(iou_n, 1),
    }


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
    )

    parser.add_argument(
        "--clean-root",
        required=True,
    )

    parser.add_argument(
        "--degradation-root",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--det-loss",
        choices=[
            "mae",
            "binomial",
        ],
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--frozen-epochs",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--validate-every",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--head-lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            torch.cuda.get_device_name(0)
        )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    transform = build_transform()

    train_ds = (
        RandomFileReliabilityDataset(
            csv_path=args.csv,
            split="train",
            clean_root=args.clean_root,
            degradation_root=
                args.degradation_root,
            transform=transform,
            seed=args.seed,
            random_per_source=True,
        )
    )

    val_ds = (
        RandomFileReliabilityDataset(
            csv_path=args.csv,
            split="val",
            clean_root=args.clean_root,
            degradation_root=
                args.degradation_root,
            transform=transform,
            seed=args.seed,
            random_per_source=False,
        )
    )

    print(
        "Train samples/epoch:",
        len(train_ds),
    )

    print(
        "Val samples:",
        len(val_ds),
    )

    print(
        "Total train exposure:",
        len(train_ds)
        * args.epochs,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = (
        MultiTaskReliabilityModel(
            pretrained=True
        )
        .to(device)
    )

    model.freeze_backbone()

    optimizer = AdamW(
        model.head_parameters(),
        lr=args.head_lr,
        weight_decay=
            args.weight_decay,
    )

    config = vars(args).copy()

    config[
        "train_samples_per_epoch"
    ] = len(train_ds)

    config[
        "total_sample_exposure"
    ] = (
        len(train_ds)
        * args.epochs
    )

    with open(
        output_dir / "config.json",
        "w",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    history = []
    best_val = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # source당 새 random state
        train_ds.set_epoch(
            epoch - 1
        )

        # ----------------------------------------
        # Exposure-matched unfreeze
        # ----------------------------------------
        if (
            epoch
            == args.frozen_epochs + 1
        ):

            print(
                "\n>>> BACKBONE UNFREEZE"
            )

            model.unfreeze_backbone()

            optimizer = AdamW(
                [
                    {
                        "params":
                            model.backbone_parameters(),
                        "lr":
                            args.backbone_lr,
                    },
                    {
                        "params":
                            model.head_parameters(),
                        "lr":
                            args.head_lr,
                    },
                ],
                weight_decay=
                    args.weight_decay,
            )

        stage = (
            "head_only"
            if epoch
            <= args.frozen_epochs
            else "fine_tuning"
        )

        logical_full_epoch = (
            epoch / 16.0
        )

        print(
            f"\nEpoch "
            f"{epoch}/{args.epochs}"
            f" | equivalent full "
            f"epoch={logical_full_epoch:.2f}"
            f" | {stage}"
            f" | {args.det_loss}"
        )

        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.det_loss,
        )

        row = {
            "epoch": epoch,
            "equivalent_full_epoch":
                logical_full_epoch,

            "stage": stage,

            "train_total_loss":
                train_losses[0],

            "train_det_loss":
                train_losses[1],

            "train_conf_loss":
                train_losses[2],

            "train_iou_loss":
                train_losses[3],

            "val_det_mae":
                np.nan,

            "val_conf_mae":
                np.nan,

            "val_iou_mae":
                np.nan,
        }

        # ----------------------------------------
        # Validate only every 16 random epochs
        # ----------------------------------------
        if (
            epoch
            % args.validate_every
            == 0
            or epoch == args.epochs
        ):

            metrics = evaluate(
                model,
                val_loader,
                device,
            )

            row.update(metrics)

            print(
                "Val Detection MAE:",
                f"{metrics['val_det_mae']:.4f}"
            )

            print(
                "Val Conf MAE:",
                f"{metrics['val_conf_mae']:.4f}"
            )

            print(
                "Val IoU MAE:",
                f"{metrics['val_iou_mae']:.4f}"
            )

            if (
                metrics["val_det_mae"]
                < best_val
            ):

                best_val = (
                    metrics[
                        "val_det_mae"
                    ]
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict":
                            model.state_dict(),
                        "val_det_mae":
                            best_val,
                        "det_loss_type":
                            args.det_loss,
                    },

                    output_dir
                    / "best_model.pt",
                )

                print(
                    ">>> Best model saved"
                )

        history.append(row)

        pd.DataFrame(
            history
        ).to_csv(
            output_dir
            / "history.csv",
            index=False,
        )

    print("\nFINISHED")
    print(
        "Best Val Detection MAE:",
        best_val,
    )


if __name__ == "__main__":
    main()