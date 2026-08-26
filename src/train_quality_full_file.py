import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.train_quality_random_file import (
    AllStateDataset,
    QualityWatchdog,
    evaluate,
    seed_everything,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)

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
        default=15,
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

    df = pd.read_csv(args.csv)

    train_df = df[
        df["split"].str.lower() == "train"
    ].copy()

    val_df = df[
        df["split"].str.lower() == "val"
    ].copy()

    print("\n=== DATASET CHECK ===")
    print("[TRAIN] sources =", train_df["image_id"].nunique())
    print("[TRAIN] rows    =", len(train_df))
    print("[VAL] sources   =", val_df["image_id"].nunique())
    print("[VAL] rows      =", len(val_df))

    # Full: 모든 16 states 사용
    train_dataset = AllStateDataset(
        dataframe=train_df,
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
    )

    val_dataset = AllStateDataset(
        dataframe=val_df,
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
    )

    print("\nTrain samples/epoch:", len(train_dataset))
    print("Val samples:", len(val_dataset))

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

    model = QualityWatchdog().to(device)

    # Epoch 1~2 freeze
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "target": "detection_quality_retention",
        "architecture": "MobileNetV3-Small",
        "sampling": "full 16 states per source",
        "loss": "MAE",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "freeze_epochs": "1-2",
        "unfreeze_epoch": 3,
        "backbone_lr": 1e-4,
        "head_lr": 1e-3,
    }

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(config, f, indent=2)

    history = []
    best_mae = float("inf")

    for epoch in range(1, args.epochs + 1):

        if epoch == 3:
            print("\n>>> UNFREEZE BACKBONE AT EPOCH 3 <<<\n")
            model.unfreeze_backbone()

        model.train()

        # backbone frozen일 때 BN statistics도 고정
        if epoch <= 2:
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

            optimizer.zero_grad(set_to_none=True)

            preds = model(images)

            loss = criterion(preds, targets)

            loss.backward()
            optimizer.step()

            n = targets.numel()

            total_abs_error += loss.item() * n
            total_count += n

        train_mae = total_abs_error / total_count

        val_mae, val_rmse = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs}"
            f" | Train MAE {train_mae:.6f}"
            f" | Val MAE {val_mae:.6f}"
            f" | Val RMSE {val_rmse:.6f}"
        )

        row = {
            "epoch": epoch,
            "train_mae": train_mae,
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_mae": val_mae,
            "val_rmse": val_rmse,
        }

        # exposure-matched 비교용 epoch3 보존
        if epoch == 3:
            torch.save(
                checkpoint,
                output_dir / "checkpoint_epoch3.pt",
            )

        if val_mae < best_mae:
            best_mae = val_mae

            torch.save(
                checkpoint,
                output_dir / "best_model.pt",
            )

            print(
                f"           NEW BEST VAL MAE = {best_mae:.6f}"
            )

    print("\n==============================")
    print("TRAINING FINISHED")
    print("==============================")
    print("Best Val MAE:", best_mae)
    print("Output:", output_dir)


if __name__ == "__main__":
    main()