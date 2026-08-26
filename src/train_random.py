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

from src.reliability_dataset_random import RandomReliabilityDataset
from src.model_multitask import MultiTaskReliabilityModel


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
    clean_reference_count,
    retained_count,
    det_loss_type,
):
    """
    outputs:
        [:, 0] Detection Retention prediction
        [:, 1] Confidence Change prediction
        [:, 2] IoU Change prediction

    targets:
        [:, 0] Rdet = k / K
        [:, 1] Delta Confidence
        [:, 2] Delta IoU

    K = clean_reference_count
    k = retained_count

    det_loss_type:
        "mae"
            기존 방식

        "binomial"
            K-aware Binomial Negative Log-Likelihood

            K가 큰 frame은 더 많은 object observation을
            포함하므로 더 강한 supervision을 제공한다.

            전체 loss scale이 K에 따라 무한히 커지지 않도록
            batch 전체 object count로 normalize한다.
    """

    pred_det = outputs[:, 0]
    pred_conf = outputs[:, 1]
    pred_iou = outputs[:, 2]

    target_det = targets[:, 0]
    target_conf = targets[:, 1]
    target_iou = targets[:, 2]

    # ========================================================
    # 1. Detection Retention
    # ========================================================

    if det_loss_type == "mae":

        # 기존 baseline과 동일
        det_loss = F.l1_loss(
            pred_det,
            target_det,
        )

    elif det_loss_type == "binomial":

        K = clean_reference_count.float()
        k = retained_count.float()

        # 현재 label dataset에서는 K > 0이어야 한다.
        if torch.any(K <= 0):
            raise ValueError(
                "Binomial loss requires K > 0."
            )

        # log(0) 방지
        eps = 1e-6

        p = pred_det.clamp(
            min=eps,
            max=1.0 - eps,
        )

        # ----------------------------------------------------
        # Frame i:
        #
        # L_i =
        # - [k_i log(p_i)
        #    + (K_i-k_i) log(1-p_i)]
        #
        # K가 큰 frame은 자연스럽게 더 큰 정보량을 가짐.
        # ----------------------------------------------------

        binomial_nll = -(
            k * torch.log(p)
            +
            (K - k) * torch.log(1.0 - p)
        )

        # ----------------------------------------------------
        # raw NLL을 그냥 batch mean 하면
        # Detection loss scale 자체가 매우 커질 수 있음.
        #
        # 그래서 전체 object observation 수로 normalize.
        #
        # 결과적으로:
        # object 하나당 평균 NLL
        #
        # 단, K=10 frame은 K=1 frame보다
        # 여전히 10배 많은 observation을 제공함.
        # ----------------------------------------------------

        det_loss = (
            binomial_nll.sum()
            /
            K.sum().clamp(min=1.0)
        )

    else:
        raise ValueError(
            f"Unknown det_loss_type: "
            f"{det_loss_type}"
        )

    # ========================================================
    # 2. Confidence / IoU
    #
    # k == 0인 sample에서는 정의되지 않으므로 mask
    # ========================================================

    mask = quality_valid_mask.float()

    valid_count = mask.sum().clamp(
        min=1.0
    )

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

    # ========================================================
    # 3. Multi-task total loss
    # ========================================================

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
# One random sampling pass
# ============================================================

def train_one_sampling_pass(
    model,
    loader,
    optimizer,
    device,
    det_loss_type,
    description,
):
    """
    RandomReliabilityDataset에서:

        source image 1장
            ↓
        condition 1개 랜덤 선택

    을 전체 source에 대해 한 번 수행.
    """

    model.train()

    total_sum = 0.0
    det_sum = 0.0
    conf_sum = 0.0
    iou_sum = 0.0

    batch_count = 0

    progress = tqdm(
        loader,
        desc=description,
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

        K = batch[
            "clean_reference_count"
        ].to(
            device,
            non_blocking=True,
        )

        k = batch[
            "retained_count"
        ].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad()

        outputs = model(images)

        losses = calculate_loss(
            outputs=outputs,
            targets=targets,
            quality_valid_mask=quality_mask,
            clean_reference_count=K,
            retained_count=k,
            det_loss_type=det_loss_type,
        )

        losses["total"].backward()

        optimizer.step()

        total_sum += losses[
            "total"
        ].item()

        det_sum += losses[
            "det"
        ].item()

        conf_sum += losses[
            "conf"
        ].item()

        iou_sum += losses[
            "iou"
        ].item()

        batch_count += 1

        progress.set_postfix(
            loss=f"{losses['total'].item():.4f}"
        )

    return {
        "total": total_sum / batch_count,
        "det": det_sum / batch_count,
        "conf": conf_sum / batch_count,
        "iou": iou_sum / batch_count,
    }


# ============================================================
# Training block
#
# 기존 full dataset:
#
#   source 3331
#   × 16 states
#   ≈ 53,296 samples / epoch
#
# random dataset:
#
#   source 3331
#   × 1 state
#   = 3331 samples / sampling pass
#
# 따라서 16 sampling passes를 묶으면
# 기존 1 epoch과 거의 동일한 sample exposure.
# ============================================================

def train_one_epoch(
    model,
    loader,
    train_dataset,
    optimizer,
    device,
    det_loss_type,
    passes_per_epoch,
    epoch,
):
    total_sum = 0.0
    det_sum = 0.0
    conf_sum = 0.0
    iou_sum = 0.0

    for pass_idx in range(
        passes_per_epoch
    ):

        # ----------------------------------------------------
        # global sampling pass 번호
        #
        # 매 pass마다 source별로 다른 condition 선택
        # ----------------------------------------------------

        global_pass = (
            (epoch - 1)
            * passes_per_epoch
            + pass_idx
        )

        train_dataset.set_epoch(
            global_pass
        )

        metrics = train_one_sampling_pass(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            det_loss_type=det_loss_type,
            description=(
                f"Train "
                f"{pass_idx + 1}/"
                f"{passes_per_epoch}"
            ),
        )

        total_sum += metrics["total"]
        det_sum += metrics["det"]
        conf_sum += metrics["conf"]
        iou_sum += metrics["iou"]

    return {
        "train_total_loss":
            total_sum / passes_per_epoch,

        "train_det_loss":
            det_sum / passes_per_epoch,

        "train_conf_loss":
            conf_sum / passes_per_epoch,

        "train_iou_loss":
            iou_sum / passes_per_epoch,
    }


# ============================================================
# Validation
#
# IMPORTANT:
# Validation에서는 random sampling 사용하지 않음.
# 모든 Clean + 15 degradation을 평가.
#
# 평가 지표는 loss 종류와 관계없이 항상
# Rdet 기준 MAE를 사용.
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

        # ----------------------------------------------------
        # Detection Retention MAE
        # ----------------------------------------------------

        det_abs_sum += torch.abs(
            outputs[:, 0]
            - targets[:, 0]
        ).sum().item()

        det_count += len(images)

        # ----------------------------------------------------
        # Confidence / IoU MAE
        # ----------------------------------------------------

        if mask.any():

            conf_abs_sum += torch.abs(
                outputs[mask, 1]
                - targets[mask, 1]
            ).sum().item()

            conf_count += (
                mask.sum().item()
            )

            iou_abs_sum += torch.abs(
                outputs[mask, 2]
                - targets[mask, 2]
            ).sum().item()

            iou_count += (
                mask.sum().item()
            )

    return {
        "val_det_mae":
            det_abs_sum
            / max(det_count, 1),

        "val_conf_mae":
            conf_abs_sum
            / max(conf_count, 1),

        "val_iou_mae":
            iou_abs_sum
            / max(iou_count, 1),
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
        "--output-dir",
        required=True,
    )

    # --------------------------------------------------------
    # Detection loss 선택
    # --------------------------------------------------------

    parser.add_argument(
        "--det-loss",
        choices=[
            "mae",
            "binomial",
        ],
        default="mae",
    )

    # --------------------------------------------------------
    # Baseline과 비교하기 위한 logical epoch
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
    )

    # --------------------------------------------------------
    # source당 한 상태를 고르는 sampling pass를
    # 몇 번 수행할 것인지
    #
    # 16으로 하면 기존 full dataset과
    # sample exposure가 거의 동일.
    # --------------------------------------------------------

    parser.add_argument(
        "--passes-per-epoch",
        type=int,
        default=16,
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

    # ========================================================
    # Setup
    # ========================================================

    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("=" * 70)
    print("RANDOM RELIABILITY TRAINING")
    print("=" * 70)

    print(
        "Device              :",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU                 :",
            torch.cuda.get_device_name(0),
        )

    print(
        "Detection loss      :",
        args.det_loss,
    )

    print(
        "Passes / epoch      :",
        args.passes_per_epoch,
    )

    # ========================================================
    # Dataset
    # ========================================================

    transform = build_transform()

    train_dataset = RandomReliabilityDataset(
        csv_path=args.csv,
        split="train",
        clean_root=args.clean_root,
        transform=transform,
        seed=args.seed,
        random_per_source=True,
    )

    val_dataset = RandomReliabilityDataset(
        csv_path=args.csv,
        split="val",
        clean_root=args.clean_root,
        transform=transform,
        seed=args.seed,
        random_per_source=False,
    )

    print(
        "Train sources/pass  :",
        len(train_dataset),
    )

    print(
        "Val samples         :",
        len(val_dataset),
    )

    print(
        "Approx samples/epoch:",
        (
            len(train_dataset)
            * args.passes_per_epoch
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # ========================================================
    # Model
    # ========================================================

    model = MultiTaskReliabilityModel(
        pretrained=True,
    ).to(device)

    # --------------------------------------------------------
    # Stage 1:
    # Head only
    # --------------------------------------------------------

    model.freeze_backbone()

    optimizer = AdamW(
        model.head_parameters(),
        lr=args.head_lr,
        weight_decay=args.weight_decay,
    )

    # ========================================================
    # Save config
    # ========================================================

    config = vars(args).copy()

    config[
        "train_sources_per_pass"
    ] = len(train_dataset)

    config[
        "estimated_samples_per_epoch"
    ] = (
        len(train_dataset)
        * args.passes_per_epoch
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

    history = []

    best_val_det_mae = float("inf")

    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        # ----------------------------------------------------
        # Fixed unfreeze
        #
        # baseline과 동일:
        # Epoch 1~2 head only
        # Epoch 3~   fine-tuning
        # ----------------------------------------------------

        if (
            epoch
            == args.frozen_epochs + 1
        ):

            print()
            print(
                ">>> Backbone UNFREEZE"
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
            if epoch <= args.frozen_epochs
            else "fine_tuning"
        )

        print()
        print("=" * 70)

        print(
            f"Epoch "
            f"{epoch}/"
            f"{args.epochs}"
            f" | {stage}"
            f" | det_loss="
            f"{args.det_loss}"
        )

        print("=" * 70)

        # ----------------------------------------------------
        # Random training
        # ----------------------------------------------------

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            train_dataset=train_dataset,
            optimizer=optimizer,
            device=device,
            det_loss_type=args.det_loss,
            passes_per_epoch=
                args.passes_per_epoch,
            epoch=epoch,
        )

        # ----------------------------------------------------
        # Fixed validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        row = {
            "epoch": epoch,
            "stage": stage,
            "det_loss_type":
                args.det_loss,

            "passes_per_epoch":
                args.passes_per_epoch,

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
            output_dir
            / "history.csv",
            index=False,
        )

        print()
        print(
            f"Train total loss : "
            f"{train_metrics['train_total_loss']:.4f}"
        )

        print(
            f"Train Det loss   : "
            f"{train_metrics['train_det_loss']:.4f}"
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

        # ====================================================
        # Best model
        #
        # loss 종류와 관계없이
        # 동일한 Val Detection MAE로 선택
        # ====================================================

        if (
            val_metrics["val_det_mae"]
            < best_val_det_mae
        ):

            best_val_det_mae = (
                val_metrics[
                    "val_det_mae"
                ]
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

                    "det_loss_type":
                        args.det_loss,

                    "passes_per_epoch":
                        args.passes_per_epoch,
                },

                output_dir
                / "best_model.pt",
            )

            print(
                ">>> Best model saved"
            )

    # ========================================================
    # Finish
    # ========================================================

    print()
    print("=" * 70)
    print("TRAINING FINISHED")

    print(
        "Detection loss:",
        args.det_loss,
    )

    print(
        "Best Validation Detection MAE:",
        best_val_det_mae,
    )

    print(
        "Results:",
        output_dir,
    )

    print("=" * 70)


if __name__ == "__main__":
    main()