import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from reliability_dataset import ReliabilityDataset
from model_multitask import MultiTaskReliabilityModel

# Fixed 실험과 완전히 동일한 함수 재사용
from train_fixed_unfreeze import (
    set_seed,
    build_transform,
    train_one_epoch,
    evaluate,
)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--degradation-root", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Adaptive unfreeze criterion
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.002)

    args = parser.parse_args()

    # ========================================================
    # Setup
    # ========================================================

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ========================================================
    # Dataset
    # ========================================================

    transform = build_transform()

    train_dataset = ReliabilityDataset(
        csv_path=args.csv,
        split="train",
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
        transform=transform,
    )

    val_dataset = ReliabilityDataset(
        csv_path=args.csv,
        split="val",
        clean_root=args.clean_root,
        degradation_root=args.degradation_root,
        transform=transform,
    )

    print("Train samples:", len(train_dataset))
    print("Val samples:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ========================================================
    # Model
    # ========================================================

    model = MultiTaskReliabilityModel(
        pretrained=True
    ).to(device)

    # 처음에는 backbone freeze
    model.freeze_backbone()

    optimizer = AdamW(
        model.head_parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # Adaptive state
    # ========================================================

    backbone_unfrozen = False
    unfreeze_epoch = None

    # plateau 판단용
    best_plateau_metric = float("inf")
    no_improvement_count = 0

    # checkpoint 선택용
    best_val_det_mae = float("inf")

    history = []

    # ========================================================
    # Config
    # ========================================================

    config = vars(args).copy()
    config["unfreeze_strategy"] = "adaptive_plateau"

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(1, args.epochs + 1):

        stage = (
            "fine_tuning"
            if backbone_unfrozen
            else "head_only"
        )

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | {stage}"
        )
        print("=" * 70)

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        current_det_mae = val_metrics[
            "val_det_mae"
        ]

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if current_det_mae < best_val_det_mae:

            best_val_det_mae = current_det_mae

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "val_det_mae":
                        current_det_mae,

                    "val_conf_mae":
                        val_metrics[
                            "val_conf_mae"
                        ],

                    "val_iou_mae":
                        val_metrics[
                            "val_iou_mae"
                        ],

                    "unfreeze_epoch":
                        unfreeze_epoch,
                },
                output_dir / "best_model.pt",
            )

            print(">>> Best model saved")

        # ----------------------------------------------------
        # Adaptive plateau 판단
        #
        # Detection Retention MAE가 min_delta 이상
        # 개선되어야 improvement로 인정
        # ----------------------------------------------------

        if not backbone_unfrozen:

            improvement = (
                best_plateau_metric
                - current_det_mae
            )

            if improvement >= args.min_delta:

                best_plateau_metric = (
                    current_det_mae
                )

                no_improvement_count = 0

                print(
                    f">>> Improvement detected: "
                    f"{improvement:.6f}"
                )

            else:

                no_improvement_count += 1

                print(
                    f">>> Plateau count: "
                    f"{no_improvement_count}"
                    f"/{args.patience}"
                )

            # -----------------------------------------------
            # patience 도달 → backbone unfreeze
            # 다음 epoch부터 fine-tuning
            # -----------------------------------------------

            if (
                no_improvement_count
                >= args.patience
            ):

                backbone_unfrozen = True
                unfreeze_epoch = epoch + 1

                print()
                print(
                    ">>> Validation Detection MAE plateau"
                )

                print(
                    f">>> Backbone will UNFREEZE "
                    f"at epoch {unfreeze_epoch}"
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

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        row = {
            "epoch": epoch,
            "stage": stage,

            **train_metrics,
            **val_metrics,

            "head_lr":
                args.head_lr,

            "backbone_lr":
                args.backbone_lr
                if backbone_unfrozen
                else 0.0,

            "plateau_count":
                no_improvement_count,

            "unfreeze_epoch":
                unfreeze_epoch,
        }

        history.append(row)

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        print(
            f"Train total loss : "
            f"{train_metrics['train_total_loss']:.4f}"
        )

        print(
            f"Val Detection MAE: "
            f"{val_metrics['val_det_mae']:.4f}"
        )

        print(
            f"Val Conf MAE     : "
            f"{val_metrics['val_conf_mae']:.4f}"
        )

        print(
            f"Val IoU MAE      : "
            f"{val_metrics['val_iou_mae']:.4f}"
        )

    # ========================================================
    # Training finished
    # ========================================================

    config["actual_unfreeze_epoch"] = (
        unfreeze_epoch
    )

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 70)
    print("ADAPTIVE TRAINING FINISHED")
    print(
        "Best Validation Detection MAE:",
        best_val_det_mae,
    )
    print(
        "Adaptive Unfreeze Epoch:",
        unfreeze_epoch,
    )
    print(
        "Results:",
        output_dir,
    )


if __name__ == "__main__":
    main()