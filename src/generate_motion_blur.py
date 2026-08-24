import argparse
from pathlib import Path

import cv2
import numpy as np


BLUR_LENGTHS = [3, 7, 11, 15, 21]
ANGLES = [0, 45, 90, 135]


def make_motion_psf(length: int, angle: float) -> np.ndarray:
    """
    Create a normalized linear motion blur PSF.

    length : approximate motion path length in pixels
    angle  : motion direction in degrees
    """

    if length < 3 or length % 2 == 0:
        raise ValueError("length must be an odd integer >= 3")

    kernel = np.zeros((length, length), dtype=np.float32)

    center = length // 2
    radius = length // 2

    theta = np.deg2rad(angle)

    dx = int(round(radius * np.cos(theta)))
    dy = int(round(radius * np.sin(theta)))

    pt1 = (center - dx, center - dy)
    pt2 = (center + dx, center + dy)

    cv2.line(
        kernel,
        pt1,
        pt2,
        color=1.0,
        thickness=1,
    )

    kernel /= kernel.sum()

    return kernel


def apply_motion_blur(
    image: np.ndarray,
    length: int,
    angle: float,
) -> np.ndarray:

    kernel = make_motion_psf(length, angle)

    blurred = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    return blurred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input image path",
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures/motion_blur_preview",
        help="Directory to save preview images",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    for length in BLUR_LENGTHS:
        for angle in ANGLES:

            blurred = apply_motion_blur(
                image,
                length=length,
                angle=angle,
            )

            output_path = (
                output_dir
                / f"motion_d{length}_angle{angle}.jpg"
            )

            cv2.imwrite(
                str(output_path),
                blurred,
                [cv2.IMWRITE_JPEG_QUALITY, 100],
            )

            print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()