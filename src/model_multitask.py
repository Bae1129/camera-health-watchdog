import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
)


class RegressionHead(nn.Module):
    """
    Shared backbone feature를 하나의 regression target으로 변환.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.layers(x)


class MultiTaskReliabilityModel(nn.Module):
    """
    Input image
        ↓
    ImageNet pretrained MobileNetV3-Small backbone
        ↓
    Shared feature vector
        ├─ Detection Retention Head
        ├─ Confidence Change Head
        └─ IoU Change Head

    Output order:
        [Detection Retention, Confidence Change, IoU Change]
    """

    def __init__(
        self,
        pretrained: bool = True,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        # -----------------------------------------------------
        # 1. Pretrained MobileNetV3-Small
        # -----------------------------------------------------
        weights = (
            MobileNet_V3_Small_Weights.DEFAULT
            if pretrained
            else None
        )

        base_model = mobilenet_v3_small(
            weights=weights
        )

        # ImageNet classifier는 사용하지 않음.
        self.backbone = base_model.features
        self.avgpool = base_model.avgpool

        # MobileNetV3-Small 최종 feature dimension
        feature_dim = 576

        # -----------------------------------------------------
        # 2. Independent Regression Heads
        # -----------------------------------------------------

        # Detection Retention: 0 ~ 1
        self.detection_head = RegressionHead(
            feature_dim,
            hidden_dim,
            dropout,
        )

        # Confidence Change: +/- 값
        self.confidence_head = RegressionHead(
            feature_dim,
            hidden_dim,
            dropout,
        )

        # IoU Change: +/- 값
        self.iou_head = RegressionHead(
            feature_dim,
            hidden_dim,
            dropout,
        )

        self.backbone_frozen = False

    def extract_features(self, x):
        """
        Shared CNN feature extraction.
        """
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward(self, x):

        features = self.extract_features(x)

        # -----------------------------------------------------
        # Detection Retention
        # 항상 0~1 범위이므로 Sigmoid
        # -----------------------------------------------------
        detection = torch.sigmoid(
            self.detection_head(features)
        )

        # -----------------------------------------------------
        # Confidence / IoU Change
        # 음수/양수 모두 가능 → Linear output 유지
        # -----------------------------------------------------
        confidence = self.confidence_head(features)
        iou = self.iou_head(features)

        # [batch, 3]
        outputs = torch.cat(
            [
                detection,
                confidence,
                iou,
            ],
            dim=1,
        )

        return outputs

    # =========================================================
    # Transfer-learning control
    # =========================================================

    def freeze_backbone(self):
        """
        Head-only 학습 단계.
        Backbone parameter update 중지.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone_frozen = True

        # BatchNorm running statistics도 고정
        self.backbone.eval()

    def unfreeze_backbone(self):
        """
        Fine-tuning 단계.
        Backbone parameter update 허용.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

        self.backbone_frozen = False

        self.backbone.train()

    def train(self, mode: bool = True):
        """
        model.train() 호출 시에도 backbone이 freeze 상태라면
        BatchNorm 통계가 변하지 않도록 eval 상태를 유지.
        """
        super().train(mode)

        if self.backbone_frozen:
            self.backbone.eval()

        return self

    # =========================================================
    # Optimizer parameter groups용
    # =========================================================

    def backbone_parameters(self):
        return self.backbone.parameters()

    def head_parameters(self):
        return (
            list(self.detection_head.parameters())
            + list(self.confidence_head.parameters())
            + list(self.iou_head.parameters())
        )


if __name__ == "__main__":

    # 간단한 구조 및 shape 검증
    model = MultiTaskReliabilityModel(
        pretrained=False
    )

    dummy = torch.randn(
        2,
        3,
        224,
        224,
    )

    output = model(dummy)

    print(model)
    print("\nInput :", dummy.shape)
    print("Output:", output.shape)

    print(
        "\nOutput order:",
        "[Detection Retention, Confidence Change, IoU Change]"
    )