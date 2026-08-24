import argparse
from pathlib import Path

import cv2
import numpy as np


NOISE_SIGMAS = [5, 10, 15, 20, 25]


def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int = 42,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise to an 8-bit image.

    noise ~ N(0, sigma^2)

    sigma : noise standard deviation in pixel intensity (0~255)
    seed  : random seed for reproducibility
    """

    if sigma <= 0:
        raise ValueError("sigma must be greater than 0")

    rng = np.random.default_rng(seed)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape,
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    # Pixel range 유지
    noisy = np.clip(noisy, 0, 255)

    return noisy.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Input image path",
    )

    parser.add_argument(
        "--output-dir",
        default="results/figures/gaussian_noise_preview",
        help="Directory to save preview images",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    for sigma in NOISE_SIGMAS:

        noisy = apply_gaussian_noise(
            image,
            sigma=sigma,
            seed=args.seed,
        )

        output_path = output_dir / f"gaussian_sigma{sigma}.jpg"

        cv2.imwrite(
            str(output_path),
            noisy,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()