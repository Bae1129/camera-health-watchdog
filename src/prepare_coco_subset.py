import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=str,
        default="data/splits/pilot_500_ids.txt"
    )
    args = parser.parse_args()

    # COCO annotation JSON 불러오기
    with open(args.ann, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # COCO val 이미지 ID 전체
    image_ids = sorted([img["id"] for img in coco["images"]])

    if args.num_images > len(image_ids):
        raise ValueError(
            f"Requested {args.num_images} images, "
            f"but dataset contains only {len(image_ids)}."
        )

    # 같은 seed를 사용하면 항상 같은 500장이 선택됨
    random.seed(args.seed)
    selected_ids = sorted(
        random.sample(image_ids, args.num_images)
    )

    # 저장 경로 생성
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 이미지 ID 저장
    with open(out_path, "w", encoding="utf-8") as f:
        for image_id in selected_ids:
            f.write(f"{image_id}\n")

    print("=== COCO Pilot Subset ===")
    print(f"Total COCO images : {len(image_ids)}")
    print(f"Selected images   : {len(selected_ids)}")
    print(f"Random seed       : {args.seed}")
    print(f"Saved to          : {out_path}")


if __name__ == "__main__":
    main()