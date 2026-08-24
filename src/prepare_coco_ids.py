import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create a reproducible COCO image ID list."
    )

    parser.add_argument(
        "--ann",
        type=str,
        required=True,
        help="Path to COCO annotation JSON"
    )

    # --num-images와 --all 중 반드시 하나만 선택
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--num-images",
        type=int,
        help="Randomly select N images"
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Use all images"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --num-images"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output txt file path"
    )

    args = parser.parse_args()

    # --------------------------------------------------
    # 1. COCO annotation 불러오기
    # --------------------------------------------------
    ann_path = Path(args.ann)

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # 모든 COCO image ID
    image_ids = sorted(
        image["id"] for image in coco["images"]
    )

    total_images = len(image_ids)

    # --------------------------------------------------
    # 2. Pilot 또는 Full 선택
    # --------------------------------------------------
    if args.all:
        selected_ids = image_ids
        mode = "FULL"

    else:
        if args.num_images <= 0:
            raise ValueError("--num-images must be greater than 0.")

        if args.num_images > total_images:
            raise ValueError(
                f"Requested {args.num_images} images, "
                f"but dataset contains only {total_images}."
            )

        rng = random.Random(args.seed)

        selected_ids = rng.sample(
            image_ids,
            args.num_images
        )

        # 파일을 보기 쉽고 일관되게 만들기 위해 정렬
        selected_ids = sorted(selected_ids)

        mode = "PILOT"

    # --------------------------------------------------
    # 3. ID 목록 저장
    # --------------------------------------------------
    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(output_path, "w", encoding="utf-8") as f:
        for image_id in selected_ids:
            f.write(f"{image_id}\n")

    # --------------------------------------------------
    # 4. 결과 출력
    # --------------------------------------------------
    print("=== COCO Image ID Preparation ===")
    print(f"Mode              : {mode}")
    print(f"Total COCO images : {total_images}")
    print(f"Selected images   : {len(selected_ids)}")

    if not args.all:
        print(f"Random seed       : {args.seed}")

    print(f"Saved to          : {output_path}")


if __name__ == "__main__":
    main()