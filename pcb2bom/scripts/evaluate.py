"""Evaluation script for pcb2bom.

Measures and reports REAL performance metrics across a labeled set of PCB photos:
- Detection Precision, Recall, and F1-score (at IoU threshold >= 0.4)
- OCR Character/String Accuracy on IC markings
- Catalog Identification Accuracy (fraction of ICs matched to expected MPN)

Usage:
    python scripts/evaluate.py [--annotations data/annotations/ground_truth.json] [--images data/sample_images]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure pcb2bom package is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rapidfuzz import fuzz

from pcb2bom.detect import calculate_iou, detect_components
from pcb2bom.lookup import run_lookup_pipeline
from pcb2bom.models import BoundingBox, Component
from pcb2bom.ocr import run_ocr_pipeline
from pcb2bom.preprocess import load_image, preprocess_image

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate")


@dataclass
class EvaluationMetrics:
    """Real measured evaluation numbers."""

    total_ground_truth_components: int = 0
    total_detected_components: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0

    total_ics_ground_truth: int = 0
    ics_correctly_ocred: int = 0
    ocr_accuracy: float = 0.0

    ics_correctly_identified: int = 0
    identification_accuracy: float = 0.0


def evaluate_dataset(
    annotations_path: Path,
    images_dir: Path,
    iou_threshold: float = 0.40,
    weights_path: Optional[str] = None,
    mock_db_path: Optional[Path] = None,
) -> EvaluationMetrics:
    """Evaluates the pipeline on a ground-truth annotated dataset.

    Args:
        annotations_path: Path to ground_truth.json.
        images_dir: Directory containing corresponding image files.
        iou_threshold: Minimum IoU overlap to count as a True Positive.
        weights_path: Optional path to YOLO weights.
        mock_db_path: Optional path to mock parts JSON.

    Returns:
        EvaluationMetrics with measured numbers.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        ground_truth_records: List[Dict[str, Any]] = json.load(f)

    metrics = EvaluationMetrics()

    for item in ground_truth_records:
        image_name = item["image"]
        img_path = images_dir / image_name
        if not img_path.is_file():
            logger.warning("Image file %s not found; skipping", img_path)
            continue

        gt_comps = item.get("components", [])
        metrics.total_ground_truth_components += len(gt_comps)

        # 1. Pipeline execution on image
        prep = preprocess_image(img_path, color_correction="clahe", fix_glare=True)
        detected = detect_components(prep.processed, weights_path=weights_path)
        detected = run_ocr_pipeline(prep.processed, detected)
        detected, _ = run_lookup_pipeline(
            detected, use_mock=True, mock_file_path=mock_db_path
        )

        metrics.total_detected_components += len(detected)

        # 2. Match detections with ground truth
        matched_gt_indices = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(detected):
            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gt_comps):
                if gt_idx in matched_gt_indices:
                    continue
                gt_box = BoundingBox.from_xyxy(*gt["bbox"])
                iou = calculate_iou(det.bbox, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_gt_indices.add(best_gt_idx)
                matched_det_indices.add(det_idx)
                metrics.true_positives += 1

                # Evaluate OCR and identification if IC
                gt_target = gt_comps[best_gt_idx]
                if gt_target.get("class", "").lower() == "ic":
                    expected_ocr = gt_target.get("expected_ocr", "").strip().upper()
                    expected_mpn = gt_target.get("expected_mpn", "").strip().upper()

                    ocr_read = det.ocr_result.cleaned_text.strip().upper() if det.ocr_result else ""
                    identified_mpn = det.lookup_result.matched_mpn.strip().upper() if det.lookup_result else ""

                    # OCR matches if ratio >= 75%
                    ocr_match = fuzz.ratio(ocr_read, expected_ocr) >= 75 or expected_ocr in ocr_read or ocr_read in expected_ocr
                    if ocr_match and ocr_read:
                        metrics.ics_correctly_ocred += 1

                    # Identification matches if matched MPN corresponds to expected
                    id_match = fuzz.ratio(identified_mpn, expected_mpn) >= 75 or expected_mpn in identified_mpn
                    if id_match:
                        metrics.ics_correctly_identified += 1

        # False positives = detections that didn't match any GT
        metrics.false_positives += len(detected) - len(matched_det_indices)
        # False negatives = GT items that weren't detected
        metrics.false_negatives += len(gt_comps) - len(matched_gt_indices)

        # Count total ICs in ground truth
        for gt in gt_comps:
            if gt.get("class", "").lower() == "ic":
                metrics.total_ics_ground_truth += 1

    # Compute aggregate statistical rates
    tp = metrics.true_positives
    fp = metrics.false_positives
    fn = metrics.false_negatives

    metrics.precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    metrics.recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics.f1_score = (
        float(2 * (metrics.precision * metrics.recall) / (metrics.precision + metrics.recall))
        if (metrics.precision + metrics.recall) > 0
        else 0.0
    )

    if metrics.total_ics_ground_truth > 0:
        metrics.ocr_accuracy = float(metrics.ics_correctly_ocred / metrics.total_ics_ground_truth)
        metrics.identification_accuracy = float(metrics.ics_correctly_identified / metrics.total_ics_ground_truth)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real measured precision/recall and identification evaluation."
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/annotations/ground_truth.json",
        help="Path to ground truth JSON annotations file",
    )
    parser.add_argument(
        "--images",
        type=str,
        default="data/sample_images",
        help="Path to directory containing sample PCB images",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.40,
        help="IoU threshold for positive detection match (default: 0.40)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional path to custom YOLO model weights",
    )
    args = parser.parse_args()

    ann_path = Path(args.annotations)
    img_dir = Path(args.images)

    if not ann_path.is_file():
        logger.error("Annotation file not found: %s", ann_path)
        return 1
    if not img_dir.is_dir():
        logger.error("Images directory not found: %s", img_dir)
        return 1

    metrics = evaluate_dataset(
        annotations_path=ann_path,
        images_dir=img_dir,
        iou_threshold=args.iou,
        weights_path=args.weights,
    )

    print("\n=======================================================")
    print("           PCB2BOM REAL EVALUATION REPORT              ")
    print("=======================================================")
    print(f"Total Ground-Truth Components : {metrics.total_ground_truth_components}")
    print(f"Total Detected Components     : {metrics.total_detected_components}")
    print(f"True Positives (TP)           : {metrics.true_positives}")
    print(f"False Positives (FP)          : {metrics.false_positives}")
    print(f"False Negatives (FN)          : {metrics.false_negatives}")
    print("-------------------------------------------------------")
    print(f"Detection Precision           : {metrics.precision:.1%} ({metrics.precision:.4f})")
    print(f"Detection Recall              : {metrics.recall:.1%} ({metrics.recall:.4f})")
    print(f"Detection F1-Score            : {metrics.f1_score:.1%} ({metrics.f1_score:.4f})")
    print("-------------------------------------------------------")
    print(f"Total ICs in Ground Truth     : {metrics.total_ics_ground_truth}")
    print(f"ICs Correctly OCR'd           : {metrics.ics_correctly_ocred}/{metrics.total_ics_ground_truth} ({metrics.ocr_accuracy:.1%})")
    print(f"ICs Correctly Identified      : {metrics.ics_correctly_identified}/{metrics.total_ics_ground_truth} ({metrics.identification_accuracy:.1%})")
    print("=======================================================\n")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
