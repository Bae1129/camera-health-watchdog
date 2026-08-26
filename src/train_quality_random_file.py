import argparse
import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# Reproducibility
# ============================================================

def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Image path mapping
# ============================================================

def resolve_image_path(row, clean_root: Path, degradation_root: Path):

    image_id = int(row["image_id"])
    filename = f"{image_id:012d}.jpg"

    condition = str(row["condition"])
    severity = str(row["severity"])

    if condition == "clean":
        return clean_root / filename

    return degradation_root / condition / severity / filename


# ============================================================
# Preprocessing
# ============================================================

def build_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


# ============================================================
# Train Dataset
#
# 한 source에서 epoch마다 16 states 중 하나만 선택
# ============================================================

class RandomStateSourceDataset(Dataset):

    def __init__(
        self,
        dataframe,
        clean_root,
        degradation_root,
        seed=42,
    ):
        self.clean_root = Path(clean_root)
        self.degradation_root = Path(degradation_root)

        self.seed = int(seed)
        self.epoch = 0

        self.transform = build_transform()

        self.groups = []

        for image_id, group in dataframe.groupby("image_id"):
            self.groups.append(
                (
                    int(image_id),
                    group.reset_index(drop=True),
                )
            )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, index):

        image_id, group = self.groups[index]

        # source + epoch에 따라 deterministic하게 state 선택
        rng = random.Random(
            self.seed
            + self.epoch * 1_000_003
            + image_id
        )

        state_idx = rng.randrange(len(group))
        row = group.iloc[state_idx]

        image_path = resolve_image_path(
            row=row,
            clean_root=self.clean_root,
            degradation_root=self.degradation_root,
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(
            float(row["detection_quality_retention"]),
            dtype=torch.float32,
        )

        return image, target


# ============================================================
# Validation Dataset
#
# Validation은 모든 16 states 사용
# ============================================================

class AllStateDataset(Dataset):

    def __init__(
        self,
        dataframe,
        clean_root,
        degradation_root,
    ):
        self.df = dataframe.reset_index(drop=True)

        self.clean_root = Path(clean_root)
        self.degradation_root = Path(degradation_root)

        self.transform = build_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = resolve_image_path(
            row=row,
            clean_root=self.clean_root,
            degradation_root=self.degradation_root,
        )

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        target = torch.tensor(
            float(row["detection_quality_retention"]),
            dtype=torch.float32,
        )

        return image, target


# ============================================================
# MobileNetV3-Small Watchdog
# ============================================================

class QualityWatchdog(nn.Module):

    def __init__(self):
        super().__init__()

        base = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT
        )

        self.features = base.features
        self.avgpool = base.avgpool

        self.head = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return self.head(x).squeeze(1)

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = True


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device):

    model.eval()

    abs_error_sum = 0.0
    squared_error_sum = 0.0
    count = 0

    for images, targets in loader:

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        preds = model(images)

        error = preds - targets

        abs_error_sum += error.abs().sum().item()
        squared_error_sum += error.pow(2).sum().item()

        count += targets.numel()

    mae = abs_error_sum / count
    rmse = (squared_error_sum / count) ** 0.5

    return mae, rmse


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
        default="/content/coco/val2017",
    )

    parser.add_argument(
        "--degradation-root",
        default="/content/degradation_full",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=48,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    df = pd.read_csv(args.csv)

    required_columns = {
        "image_id",
        "condition",
        "severity",
        "split",
        "detection_quality_retention",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing CSV columns: {sorted(missing)}"
        )

    train_df = df[
        df["split"].str.lower() == "train"
    ].copy()

    val_df = df[
        df["split"].str.lower() == "val"
    ].copy()

    print()
    print("=== DATASET CHECK ===")
    print(
        "[TRAIN] sources =",
        train_df["image_id"].nunique(),
    )
    print(
        "[TRAIN] rows    =",
        len(train_df),
    )
    print(
        "[VAL] sources   =",
        val_df["image_id"].nunique(),
    )
    print(
        "[VAL] rows      =",
        len(val_df),
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = RandomStateSourceDataset(
        dataframe=train_df,
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
        seed=args.seed,
    )

    val_dataset = AllStateDataset(
        dataframe=val_df,
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
    )

    print()
    print(
        "Train samples/epoch:",
        len(train_dataset),
    )
    print(
        "Val samples:",
        len(val_dataset),
    )

    if len(train_dataset) != train_df["image_id"].nunique():
        raise RuntimeError(
            "Train dataset must contain exactly "
            "one sample per source per epoch."
        )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = QualityWatchdog().to(device)

    model.freeze_backbone()

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": 1e-4,
            },
            {
                "params": model.head.parameters(),
                "lr": 1e-3,
            },
        ],
        weight_decay=1e-4,
    )

    criterion = nn.L1Loss()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = {
        "target": "detection_quality_retention",
        "architecture": "MobileNetV3-Small",
        "input_size": 224,
        "sampling": (
            "1 random state per source per epoch"
        ),
        "loss": "MAE",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
        "weight_decay": 1e-4,
        "freeze_epochs": "1-32",
        "unfreeze_epoch": 33,
        "validation_epochs": [16, 32, 48],
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    history = []
    best_mae = float("inf")

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(1, args.epochs + 1):

        # ----------------------------------------------------
        # Backbone fine-tuning 시작
        # ----------------------------------------------------

        if epoch == 33:

            print()
            print(
                ">>> UNFREEZE BACKBONE AT EPOCH 33 <<<"
            )
            print()

            model.unfreeze_backbone()

        train_dataset.set_epoch(epoch)

        model.train()

        # Frozen 기간에는 BN running statistics도 고정
        if epoch <= 32:
            model.features.eval()

        total_abs_error = 0.0
        total_count = 0

        for images, targets in train_loader:

            images = images.to(
                device,
                non_blocking=True,
            )

            targets = targets.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            preds = model(images)

            loss = criterion(
                preds,
                targets,
            )

            loss.backward()
            optimizer.step()

            batch_n = targets.numel()

            total_abs_error += (
                loss.item() * batch_n
            )

            total_count += batch_n

        train_mae = (
            total_abs_error / total_count
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs}"
            f" | Train MAE {train_mae:.6f}"
        )

        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": None,
            "val_rmse": None,
        }

        # ----------------------------------------------------
        # Previous random experiment와 동일한 평가 시점
        # ----------------------------------------------------

        if epoch in {16, 32, 48}:

            val_mae, val_rmse = evaluate(
                model,
                val_loader,
                device,
            )

            row["val_mae"] = val_mae
            row["val_rmse"] = val_rmse

            print(
                f"           "
                f"Val MAE {val_mae:.6f}"
                f" | Val RMSE {val_rmse:.6f}"
            )

            checkpoint = {
                "epoch": epoch,
                "model_state_dict":
                    model.state_dict(),
                "val_mae": val_mae,
                "val_rmse": val_rmse,
            }

            torch.save(
                checkpoint,
                output_dir
                / f"checkpoint_epoch{epoch}.pt",
            )

            if val_mae < best_mae:

                best_mae = val_mae

                torch.save(
                    checkpoint,
                    output_dir / "best_model.pt",
                )

                print(
                    f"           "
                    f"NEW BEST VAL MAE = "
                    f"{best_mae:.6f}"
                )

        history.append(row)

        # 매 epoch Drive에 기록
        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

    print()
    print("==============================")
    print("TRAINING FINISHED")
    print("==============================")
    print(
        "Best measured Val MAE:",
        best_mae,
    )
    print(
        "Output:",
        output_dir,
    )


if __name__ == "__main__":
    main()