import cv2
import numpy as np


# =========================================================
# Severity definitions
# =========================================================

DEFOCUS_DIAMETERS = [3, 7, 11, 15, 21]
MOTION_LENGTHS = [3, 7, 11, 15, 21]
MOTION_ANGLES = [0, 45, 90, 135]
NOISE_SIGMAS = [5, 10, 15, 20, 25]


# =========================================================
# Defocus Blur
# =========================================================

def make_disk_psf(diameter: int) -> np.ndarray:
    """
    Create a normalized disk-shaped PSF.
    """

    if diameter <= 0:
        raise ValueError("diameter must be positive.")

    if diameter % 2 == 0:
        raise ValueError("diameter must be an odd number.")

    radius = diameter // 2

    y, x = np.ogrid[
        -radius:radius + 1,
        -radius:radius + 1
    ]

    mask = (x**2 + y**2) <= radius**2

    kernel = mask.astype(np.float32)
    kernel /= kernel.sum()

    return kernel


def apply_defocus(
    image: np.ndarray,
    diameter: int,
) -> np.ndarray:
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
        borderType=cv2.BORDER_REFLECT101,
    )

    return blurred


# =========================================================
# Motion Blur
# =========================================================

def make_motion_psf(
    length: int,
    angle: float,
) -> np.ndarray:
    """
    Create a normalized linear motion blur PSF.
    """

    if length < 3 or length % 2 == 0:
        raise ValueError("length must be an odd integer >= 3")

    kernel = np.zeros(
        (length, length),
        dtype=np.float32,
    )

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
    """
    Apply linear motion blur.
    """

    kernel = make_motion_psf(
        length=length,
        angle=angle,
    )

    blurred = cv2.filter2D(
        image,
        ddepth=-1,
        kernel=kernel,
        borderType=cv2.BORDER_REFLECT101,
    )

    return blurred


# =========================================================
# Gaussian Noise
# =========================================================

def apply_gaussian_noise(
    image: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """
    Add zero-mean Gaussian noise.

    noise ~ N(0, sigma^2)
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

    noisy = np.clip(
        noisy,
        0,
        255,
    )

    return noisy.astype(np.uint8)