import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from reliability_dataset import ReliabilityDataset
from model_multitask import MultiTaskReliabilityModel


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Image preprocessing
#
# 주의:
# blur / noise / object composition 자체가 label과 연결되어 있으므로
# RandomBlur, RandomCrop, ColorJitter 같은 augmentation은 사용하지 않음.
# ============================================================

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Multi-task Loss
# ============================================================

def calculate_loss(
    outputs,
    targets,
    quality_valid_mask,
):
    """
    outputs:
        [:, 0] Detection Retention
        [:, 1] Confidence Change
        [:, 2] IoU Change

    quality_valid_mask:
        1 -> Confidence / IoU 정답 유효
        0 -> retained_count == 0, loss에서 제외
    """

    pred_det = outputs[:, 0]
    pred_conf = outputs[:, 1]
    pred_iou = outputs[:, 2]

    target_det = targets[:, 0]
    target_conf = targets[:, 1]
    target_iou = targets[:, 2]

    # --------------------------------------------------------
    # Detection Retention
    # 모든 sample에서 사용
    # --------------------------------------------------------
    det_loss = F.l1_loss(
        pred_det,
        target_det,
    )

    # --------------------------------------------------------
    # Confidence / IoU
    # surviving object가 존재하는 sample에서만 사용
    # --------------------------------------------------------
    mask = quality_valid_mask.float()

    valid_count = mask.sum().clamp(min=1.0)

    conf_error = torch.abs(
        pred_conf - target_conf
    )

    iou_error = torch.abs(
        pred_iou - target_iou
    )

    conf_loss = (
        conf_error * mask
    ).sum() / valid_count

    iou_loss = (
        iou_error * mask
    ).sum() / valid_count

    # 우선 임의 가중치 없이 동일 가중
    total_loss = (
        det_loss
        + conf_loss
        + iou_loss
    )

    return {
        "total": total_loss,
        "det": det_loss,
        "conf": conf_loss,
        "iou": iou_loss,
    }


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
):
    model.train()

    total_sum = 0.0
    det_sum = 0.0
    conf_sum = 0.0
    iou_sum = 0.0

    batch_count = 0

    progress = tqdm(
        loader,
        desc="Train",
        leave=False,
    )

    for batch in progress:

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["targets"].to(
            device,
            non_blocking=True,
        )

        quality_mask = batch[
            "quality_valid_mask"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)

        losses = calculate_loss(
            outputs,
            targets,
            quality_mask,
        )

        losses["total"].backward()

        optimizer.step()

        total_sum += losses["total"].item()
        det_sum += losses["det"].item()
        conf_sum += losses["conf"].item()
        iou_sum += losses["iou"].item()

        batch_count += 1

        progress.set_postfix(
            loss=f"{losses['total'].item():.4f}"
        )

    return {
        "train_total_loss": total_sum / batch_count,
        "train_det_loss": det_sum / batch_count,
        "train_conf_loss": conf_sum / batch_count,
        "train_iou_loss": iou_sum / batch_count,
    }


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

    det_abs_sum = 0.0
    det_count = 0

    conf_abs_sum = 0.0
    conf_count = 0

    iou_abs_sum = 0.0
    iou_count = 0

    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False,
    ):

        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        targets = batch["targets"].to(
            device,
            non_blocking=True,
        )

        mask = batch[
            "quality_valid_mask"
        ].to(
            device,
            non_blocking=True,
        ).bool()

        outputs = model(images)

        # Detection
        det_abs_sum += torch.abs(
            outputs[:, 0] - targets[:, 0]
        ).sum().item()

        det_count += len(images)

        # Confidence / IoU
        if mask.any():

            conf_abs_sum += torch.abs(
                outputs[mask, 1]
                - targets[mask, 1]
            ).sum().item()

            conf_count += mask.sum().item()

            iou_abs_sum += torch.abs(
                outputs[mask, 2]
                - targets[mask, 2]
            ).sum().item()

            iou_count += mask.sum().item()

    return {
        "val_det_mae":
            det_abs_sum / det_count,

        "val_conf_mae":
            conf_abs_sum / max(conf_count, 1),

        "val_iou_mae":
            iou_abs_sum / max(iou_count, 1),
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
        "--epochs",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--frozen-epochs",
        type=int,
        default=2,
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

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    set_seed(args.seed)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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

    print(
        "Train samples:",
        len(train_dataset),
    )

    print(
        "Val samples:",
        len(val_dataset),
    )

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MultiTaskReliabilityModel(
        pretrained=True
    ).to(device)

    # Stage 1:
    # pretrained backbone 고정
    model.freeze_backbone()

    optimizer = AdamW(
        model.head_parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Config 저장
    # --------------------------------------------------------

    with open(
        output_dir / "config.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            vars(args),
            f,
            indent=2,
            ensure_ascii=False,
        )

    history = []

    best_val_det_mae = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ====================================================
        # Fixed unfreeze
        #
        # Epoch 1~2 : head only
        # Epoch 3~   : backbone fine-tuning
        # ====================================================

        if epoch == args.frozen_epochs + 1:

            print(
                "\n>>> Backbone UNFREEZE"
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
                weight_decay=args.weight_decay,
            )

        stage = (
            "head_only"
            if epoch <= args.frozen_epochs
            else "fine_tuning"
        )

        print()
        print("=" * 70)
        print(
            f"Epoch {epoch}/{args.epochs}"
            f" | {stage}"
        )
        print("=" * 70)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "stage": stage,

            **train_metrics,
            **val_metrics,

            "head_lr":
                args.head_lr,

            "backbone_lr":
                0.0
                if stage == "head_only"
                else args.backbone_lr,
        }

        history.append(row)

        history_df = pd.DataFrame(
            history
        )

        history_df.to_csv(
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

        # ----------------------------------------------------
        # Best model:
        # 핵심 지표 = Detection Retention MAE
        # ----------------------------------------------------

        if (
            val_metrics["val_det_mae"]
            < best_val_det_mae
        ):

            best_val_det_mae = (
                val_metrics["val_det_mae"]
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),

                    "val_det_mae":
                        best_val_det_mae,

                    "val_conf_mae":
                        val_metrics[
                            "val_conf_mae"
                        ],

                    "val_iou_mae":
                        val_metrics[
                            "val_iou_mae"
                        ],
                },
                output_dir / "best_model.pt",
            )

            print(
                ">>> Best model saved"
            )

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print(
        "Best Validation Detection MAE:",
        best_val_det_mae,
    )
    print(
        "Results:",
        output_dir,
    )


if __name__ == "__main__":
    main()