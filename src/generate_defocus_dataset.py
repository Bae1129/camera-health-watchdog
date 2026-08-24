import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from generate_defocus import apply_defocus


SEVERITIES = {
    "S1_d3": 3,
    "S2_d7": 7,
    "S3_d11": 11,
    "S4_d15": 15,
    "S5_d21": 21,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic defocus dataset from COCO images."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing original COCO images"
    )

    parser.add_argument(
        "--ids",
        type=str,
        required=True,
        help="TXT file containing COCO image IDs"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for generated defocus images"
    )

    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    ids_path = Path(args.ids)
    output_root = Path(args.output_dir)

    # ----------------------------------------
    # Image ID 목록 읽기
    # ----------------------------------------
    with open(ids_path, "r", encoding="utf-8") as f:
        image_ids = [
            int(line.strip())
            for line in f
            if line.strip()
        ]

    print("=== Defocus Dataset Generation ===")
    print(f"Images     : {len(image_ids)}")
    print(f"Severities : {len(SEVERITIES)}")
    print(f"Outputs    : {len(image_ids) * len(SEVERITIES)}")

    # ----------------------------------------
    # Severity별 생성
    # ----------------------------------------
    for severity_name, diameter in SEVERITIES.items():

        output_dir = output_root / severity_name
        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        print(
            f"\nGenerating {severity_name} "
            f"(d={diameter}px)"
        )

        for image_id in tqdm(image_ids):

            filename = f"{image_id:012d}.jpg"

            input_path = image_dir / filename
            output_path = output_dir / filename

            image = cv2.imread(str(input_path))

            if image is None:
                print(f"\nWarning: failed to read {input_path}")
                continue

            blurred = apply_defocus(
                image,
                diameter
            )

            # JPEG 재압축 영향을 최소화하기 위해 quality=100
            cv2.imwrite(
                str(output_path),
                blurred,
                [cv2.IMWRITE_JPEG_QUALITY, 100]
            )

    print("\n=== Generation Complete ===")


if __name__ == "__main__":
    main()