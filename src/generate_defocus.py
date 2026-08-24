import argparse
from pathlib import Path

import cv2
import numpy as np


SEVERITIES = {
    "S0": 0,    # Clean
    "S1": 3,
    "S2": 7,
    "S3": 11,
    "S4": 15,
    "S5": 21,
}


def make_disk_psf(diameter):
    """
    Create a normalized disk-shaped PSF.
    """

    if diameter <= 0:
        raise ValueError("diameter must be positive.")

    if diameter % 2 == 0:
        raise ValueError("diameter must be an odd number.")

    radius = diameter // 2

    # 예: diameter=7이면 좌표는 -3 ~ +3
    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    # 원 내부만 True
    mask = (x**2 + y**2) <= radius**2

    # True → 1.0, False → 0.0
    kernel = mask.astype(np.float32)

    # 전체 가중치 합 = 1
    kernel /= kernel.sum()

    return kernel


def apply_defocus(image, diameter):
    """
    Apply disk-PSF defocus blur.
    """

    if diameter == 0:
        return image.copy()

    kernel = make_disk_psf(diameter)

    blurred = cv2.filter2D(
        src=image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101
    )

    return blurred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input image path"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/defocus_preview"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    print("Input image shape:", image.shape)

    for severity, diameter in SEVERITIES.items():

        output = apply_defocus(
            image,
            diameter
        )

        output_path = output_dir / (
            f"{severity}_d{diameter}.jpg"
        )

        cv2.imwrite(
            str(output_path),
            output
        )

        print(
            f"{severity}: "
            f"d={diameter}px "
            f"shape={output.shape} "
            f"-> {output_path}"
        )


if __name__ == "__main__":
    main()