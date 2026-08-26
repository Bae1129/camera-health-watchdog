%cd /content/camera-health-watchdog

from pathlib import Path
import cv2
import csv
from tqdm.auto import tqdm

from src.generate_defocus import apply_defocus
from src.generate_motion_blur import apply_motion_blur, ANGLES
from src.generate_gaussian_noise import apply_gaussian_noise


# -------------------------------------------------
# Paths
# -------------------------------------------------
IMAGE_DIR = Path("/content/coco/val2017")
OUTPUT_ROOT = Path("/content/degradation_full")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Severity settings
# -------------------------------------------------
DEFOCUS_LEVELS = {
    "S1_d3": 3,
    "S2_d7": 7,
    "S3_d11": 11,
    "S4_d15": 15,
    "S5_d21": 21,
}

MOTION_LEVELS = {
    "M1_d3": 3,
    "M2_d7": 7,
    "M3_d11": 11,
    "M4_d15": 15,
    "M5_d21": 21,
}

NOISE_LEVELS = {
    "N1_sigma5": 5,
    "N2_sigma10": 10,
    "N3_sigma15": 15,
    "N4_sigma20": 20,
    "N5_sigma25": 25,
}

# -------------------------------------------------
# Image list
# -------------------------------------------------
image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
print("COCO images:", len(image_paths))

if len(image_paths) == 0:
    raise RuntimeError("No images found in /content/coco/val2017")

# -------------------------------------------------
# Prepare output folders
# -------------------------------------------------
for folder_name in DEFOCUS_LEVELS:
    (OUTPUT_ROOT / "defocus" / folder_name).mkdir(parents=True, exist_ok=True)

for folder_name in MOTION_LEVELS:
    (OUTPUT_ROOT / "motion_blur" / folder_name).mkdir(parents=True, exist_ok=True)

for folder_name in NOISE_LEVELS:
    (OUTPUT_ROOT / "gaussian_noise" / folder_name).mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Metadata
# -------------------------------------------------
metadata_rows = []

# -------------------------------------------------
# Generate all degradations
# -------------------------------------------------
for image_path in tqdm(image_paths, desc="Generating all degradations"):

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skip unreadable image: {image_path}")
        continue

    source_id = image_path.stem
    filename = image_path.name

    # source별 고정 angle / noise seed
    angle = ANGLES[int(source_id) % len(ANGLES)]
    noise_seed = 42 + int(source_id)

    # ---------------------------
    # Defocus
    # ---------------------------
    for severity_name, diameter in DEFOCUS_LEVELS.items():

        out_dir = OUTPUT_ROOT / "defocus" / severity_name
        out_path = out_dir / filename

        degraded = apply_defocus(
            image=image,
            diameter=diameter
        )

        cv2.imwrite(
            str(out_path),
            degraded,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        metadata_rows.append([
            source_id,
            filename,
            "defocus",
            severity_name,
            diameter,
            0,
            "",
            str(out_path)
        ])

    # ---------------------------
    # Motion Blur
    # ---------------------------
    for severity_name, length in MOTION_LEVELS.items():

        out_dir = OUTPUT_ROOT / "motion_blur" / severity_name
        out_path = out_dir / filename

        degraded = apply_motion_blur(
            image=image,
            length=length,
            angle=angle
        )

        cv2.imwrite(
            str(out_path),
            degraded,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        metadata_rows.append([
            source_id,
            filename,
            "motion_blur",
            severity_name,
            length,
            angle,
            "",
            str(out_path)
        ])

    # ---------------------------
    # Gaussian Noise
    # ---------------------------
    for severity_name, sigma in NOISE_LEVELS.items():

        out_dir = OUTPUT_ROOT / "gaussian_noise" / severity_name
        out_path = out_dir / filename

        degraded = apply_gaussian_noise(
            image=image,
            sigma=sigma,
            seed=noise_seed
        )

        cv2.imwrite(
            str(out_path),
            degraded,
            [cv2.IMWRITE_JPEG_QUALITY, 100]
        )

        metadata_rows.append([
            source_id,
            filename,
            "gaussian_noise",
            severity_name,
            sigma,
            0,
            noise_seed,
            str(out_path)
        ])

# -------------------------------------------------
# Save metadata
# -------------------------------------------------
metadata_path = OUTPUT_ROOT / "degradation_metadata.csv"

with open(metadata_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "source_id",
        "filename",
        "condition",
        "severity_level",
        "parameter",
        "angle",
        "noise_seed",
        "output_path"
    ])
    writer.writerows(metadata_rows)

print("\nFinished.")
print("Metadata:", metadata_path)

# -------------------------------------------------
# Final count check
# -------------------------------------------------
defocus_count = len(list((OUTPUT_ROOT / "defocus").glob("*/*.jpg")))
motion_count = len(list((OUTPUT_ROOT / "motion_blur").glob("*/*.jpg")))
noise_count = len(list((OUTPUT_ROOT / "gaussian_noise").glob("*/*.jpg")))
total_count = defocus_count + motion_count + noise_count

print("\n=== Count Check ===")
print("Defocus      :", defocus_count)
print("Motion Blur  :", motion_count)
print("GaussianNoise:", noise_count)
print("Total        :", total_count)