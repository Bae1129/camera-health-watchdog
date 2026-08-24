import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm
from ultralytics import YOLO

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained YOLO on COCO-format dataset."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing evaluation images"
    )

    parser.add_argument(
        "--ann",
        type=str,
        required=True,
        help="COCO annotation JSON path"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model weights"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output prediction JSON path"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device: 0 for GPU, cpu for CPU"
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    annotation_path = Path(args.ann)
    prediction_path = Path(args.output)

    prediction_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # -------------------------------------------------
    # COCO Ground Truth
    # -------------------------------------------------

    coco_gt = COCO(str(annotation_path))

    image_ids = sorted(
        coco_gt.getImgIds()
    )

    print("=== YOLO COCO Evaluation ===")
    print(f"Images : {len(image_ids)}")
    print(f"Model  : {args.model}")
    print(f"imgsz  : {args.imgsz}")

    # -------------------------------------------------
    # YOLO pretrained model
    # -------------------------------------------------

    model = YOLO(args.model)

    # YOLO class index -> COCO category ID
    name_to_cat_id = {
        cat["name"]: cat["id"]
        for cat in coco_gt.loadCats(
            coco_gt.getCatIds()
        )
    }

    yolo_to_coco = {
        cls_id: name_to_cat_id[class_name]
        for cls_id, class_name
        in model.names.items()
    }

    # -------------------------------------------------
    # Inference
    # -------------------------------------------------

    predictions = []

    for image_id in tqdm(image_ids):

        image_path = (
            image_dir /
            f"{image_id:012d}.jpg"
        )

        if not image_path.exists():
            print(
                f"Warning: missing image "
                f"{image_path}"
            )
            continue

        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,

            # AP 평가를 위해 낮게 설정
            conf=0.001,

            # NMS IoU threshold
            iou=0.7,

            device=args.device,
            verbose=False
        )[0]

        boxes = result.boxes

        if boxes is None:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = (
            boxes.cls.cpu()
            .numpy()
            .astype(int)
        )

        for box, score, cls_id in zip(
            xyxy,
            scores,
            classes
        ):

            x1, y1, x2, y2 = box

            width = x2 - x1
            height = y2 - y1

            predictions.append({
                "image_id": int(image_id),

                "category_id": int(
                    yolo_to_coco[cls_id]
                ),

                # COCO bbox format:
                # [x, y, width, height]
                "bbox": [
                    float(x1),
                    float(y1),
                    float(width),
                    float(height)
                ],

                "score": float(score)
            })

    print(
        f"Total predictions: "
        f"{len(predictions)}"
    )

    # -------------------------------------------------
    # Save prediction JSON
    # -------------------------------------------------

    with open(
        prediction_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            predictions,
            f
        )

    print(
        f"Prediction saved: "
        f"{prediction_path}"
    )

    # -------------------------------------------------
    # COCO Evaluation
    # -------------------------------------------------

    coco_dt = coco_gt.loadRes(
        str(prediction_path)
    )

    coco_eval = COCOeval(
        coco_gt,
        coco_dt,
        iouType="bbox"
    )

    coco_eval.params.imgIds = image_ids

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    main()