# Camera Health Watchdog

Edge AI Camera Health Monitoring for Defocus Detection.

## Project Goal

Detect camera defocus severity and classify camera health as:

- VALID
- DEGRADED
- INVALID

The health criteria will be determined based on downstream object detection performance degradation.

## Pipeline

COCO Validation Images
→ Synthetic Defocus
→ YOLO Performance Evaluation
→ Health Threshold Definition
→ MobileNetV3-Small Training
→ ONNX Export
→ Edge Inference