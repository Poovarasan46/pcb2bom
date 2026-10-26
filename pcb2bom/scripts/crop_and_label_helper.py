"""Data preparation script: Crop detected components into categorized folders for dataset labeling.

Usage:
    python scripts/crop_and_label_helper.py --input <image_or_dir> --output <crops_dir> [--weights <model.pt>]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Ensure pcb2bom package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from pcb2bom.detect import detect_components
from pcb2bom.ocr import crop_component
from pcb2bom.preprocess import load_image

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("crop_helper")


def process_image(
    image_path: Path,
    output_dir: Path,
    weights_path: Optional[str] = None,
    manifest_writer: Optional[csv.writer] = None,
) -> int:
    """Processes a single PCB image, crops detected components, and writes crops to disk."""
    try:
        img = load_image(image_path)
    except Exception as e:
        logger.error("Failed to load '%s': %s", image_path, e)
        return 0

    components = detect_components(img, weights_path=weights_path)
    count = 0

    for comp in components:
        crop = crop_component(img, comp.bbox, margin_px=4)
        if crop.size == 0:
            continue

        class_dir = output_dir / comp.class_name.lower()
        class_dir.mkdir(parents=True, exist_ok=True)

        crop_filename = f"{image_path.stem}_{comp.id}_{comp.bbox.xmin}_{comp.bbox.ymin}.jpg"
        crop_path = class_dir / crop_filename

        cv2.imwrite(str(crop_path), crop)
        count += 1

        if manifest_writer:
            manifest_writer.writerow([
                str(image_path.name),
                comp.id,
                comp.class_name,
                comp.bbox.xmin,
                comp.bbox.ymin,
                comp.bbox.xmax,
                comp.bbox.ymax,
                f"{comp.confidence:.3f}",
                str(crop_path.relative_to(output_dir)),
            ])

    logger.info("Processed %s -> extracted %d component crops", image_path.name, count)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Crop components from PCB photos into categorized folders for manual labeling."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        type=str,
        help="Path to an individual PCB image file or a directory containing board photos",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./labeled_crops",
        help="Destination directory for categorized crops (default: ./labeled_crops)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to YOLO fine-tuned weights (.pt file)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_csv = output_dir / "crops_manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Source Image",
            "Component ID",
            "Class",
            "Xmin",
            "Ymin",
            "Xmax",
            "Ymax",
            "Confidence",
            "Crop File",
        ])

        total_crops = 0
        if input_path.is_file():
            total_crops += process_image(input_path, output_dir, args.weights, writer)
        elif input_path.is_dir():
            image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            img_files = [p for p in input_path.glob("**/*") if p.suffix.lower() in image_extensions]
            logger.info("Found %d images in %s", len(img_files), input_path)
            for img_p in img_files:
                total_crops += process_image(img_p, output_dir, args.weights, writer)
        else:
            logger.error("Provided input path does not exist: %s", input_path)
            return 1

    logger.info("All done! Extracted %d crops. Manifest saved to: %s", total_crops, manifest_csv)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
