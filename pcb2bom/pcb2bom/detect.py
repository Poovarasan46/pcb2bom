"""Component detection module for the pcb2bom pipeline.

Provides YOLO-based inference with automatic graceful fallback to an
OpenCV morphological contour and color-heuristic detector when model weights
are absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from pcb2bom.models import BoundingBox, Component

logger = logging.getLogger(__name__)

# Standard PCB component classes
PCB_CLASSES: List[str] = [
    "ic",
    "capacitor",
    "resistor",
    "connector",
    "inductor",
    "crystal",
    "diode",
    "transistor",
]

# Standard electronics reference designator prefixes
DESIGNATOR_PREFIXES: Dict[str, str] = {
    "ic": "U",
    "capacitor": "C",
    "resistor": "R",
    "connector": "J",
    "inductor": "L",
    "crystal": "Y",
    "diode": "D",
    "transistor": "Q",
}


def calculate_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculates Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First BoundingBox.
        box2: Second BoundingBox.

    Returns:
        IoU score between 0.0 and 1.0.
    """
    inter_xmin = max(box1.xmin, box2.xmin)
    inter_ymin = max(box1.ymin, box2.ymin)
    inter_xmax = min(box1.xmax, box2.xmax)
    inter_ymax = min(box1.ymax, box2.ymax)

    inter_w = max(0, inter_xmax - inter_xmin)
    inter_h = max(0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    union_area = box1.area + box2.area - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


def apply_nms(components: List[Component], iou_threshold: float = 0.45) -> List[Component]:
    """Applies Non-Maximum Suppression (NMS) to prune overlapping detections.

    Args:
        components: List of candidate Component detections.
        iou_threshold: Overlap IoU threshold above which lower-confidence boxes are pruned.

    Returns:
        Pruned list of Component detections.
    """
    if not components:
        return []

    # Sort descending by confidence
    sorted_comps = sorted(components, key=lambda c: c.confidence, reverse=True)
    selected: List[Component] = []

    for comp in sorted_comps:
        overlap = False
        for kept in selected:
            if calculate_iou(comp.bbox, kept.bbox) > iou_threshold:
                overlap = True
                break
        if not overlap:
            selected.append(comp)

    return selected


def assign_reference_designators(components: List[Component]) -> List[Component]:
    """Assigns standard schematic reference designators (e.g., U1, C1, R1)

    Components are ordered spatially (top-to-bottom, left-to-right) within
    each class before assigning numeric identifiers.

    Args:
        components: List of Component instances.

    Returns:
        List of components with assigned IDs.
    """
    class_counters: Dict[str, int] = {cls_name: 1 for cls_name in PCB_CLASSES}
    # Generic counter for unknown classes
    other_counter = 1

    # Sort primarily by ymin (row), secondarily by xmin (col)
    spatially_sorted = sorted(components, key=lambda c: (c.bbox.ymin // 30, c.bbox.xmin))

    for comp in spatially_sorted:
        cls_lower = comp.class_name.lower()
        prefix = DESIGNATOR_PREFIXES.get(cls_lower, "COMP")
        if cls_lower in class_counters:
            num = class_counters[cls_lower]
            class_counters[cls_lower] += 1
        else:
            num = other_counter
            other_counter += 1
        comp.id = f"{prefix}{num}"

    return spatially_sorted


class ContourDetector:
    """Fallback OpenCV contour-based detector for PCB components.

    Detects rectangular packages, surface mount chips, passives, and header pins
    using adaptive thresholding, morphological gradients, and dimensional heuristics.
    """

    def __init__(
        self,
        min_area: int = 150,
        max_area_fraction: float = 0.4,
        confidence: float = 0.65,
    ) -> None:
        self.min_area = min_area
        self.max_area_fraction = max_area_fraction
        self.default_confidence = confidence

    def detect(self, image: np.ndarray) -> List[Component]:
        """Detects components in image using OpenCV contours.

        Args:
            image: BGR image array.

        Returns:
            List of detected Component instances.
        """
        h, w = image.shape[:2]
        img_area = h * w
        max_area = int(img_area * self.max_area_fraction)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur slightly to smooth solder mask texture
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Morphological gradient to highlight component package edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        grad = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

        # Otsu thresholding on gradient
        _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close gaps in package outlines
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_components: List[Component] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw < 8 or ch < 8:
                continue

            # Compute aspect ratio & rectangularity
            aspect = float(cw) / float(ch) if ch > 0 else 1.0
            rect_area = cw * ch
            solidity = float(area) / float(rect_area) if rect_area > 0 else 0.0

            # Reject elongated slivers or irregular non-component shapes
            if solidity < 0.45 or aspect > 6.0 or aspect < 0.16:
                continue

            # Crop region to analyze color/texture
            crop = image[y : y + ch, x : x + cw]
            class_name, conf = self._classify_crop(crop, cw, ch, area, img_area)

            bbox = BoundingBox.from_xywh(x, y, cw, ch)
            raw_components.append(
                Component(
                    id="",  # assigned later
                    class_name=class_name,
                    bbox=bbox,
                    confidence=conf,
                )
            )

        # Prune overlapping candidate boxes
        pruned = apply_nms(raw_components, iou_threshold=0.35)
        assigned = assign_reference_designators(pruned)
        logger.info("ContourDetector found %d components after NMS", len(assigned))
        return assigned

    def _classify_crop(
        self, crop: np.ndarray, w: int, h: int, area: float, img_area: int
    ) -> Tuple[str, float]:
        """Heuristic rule-based classifier based on size, color, and aspect ratio."""
        if crop.size == 0:
            return "ic", self.default_confidence

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        std_intensity = float(np.std(gray))
        rel_area = area / float(img_area)

        # 1. Large dark bodies with internal high variance (markings/leads) -> IC
        if rel_area >= 0.003 and mean_intensity < 100:
            return "ic", min(0.85, self.default_confidence + 0.15)

        # 2. Elongated medium-sized connectors
        aspect = float(w) / float(h)
        if (aspect > 2.8 or aspect < 0.35) and rel_area >= 0.002:
            return "connector", self.default_confidence

        # 3. Small passives
        if rel_area < 0.003:
            # Check color: brownish / tan -> ceramic capacitor; dark charcoal -> resistor
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_chan, s_chan, v_chan = cv2.split(hsv)
            mean_s = np.mean(s_chan)

            if mean_s > 40 and 10 < np.mean(h_chan) < 35:
                return "capacitor", self.default_confidence
            if mean_intensity < 70:
                return "resistor", self.default_confidence
            return "capacitor", self.default_confidence - 0.05

        return "ic", self.default_confidence


class YOLODetector:
    """YOLO component detector utilizing Ultralytics weights with graceful fallback."""

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        self.weights_path = Path(weights_path) if weights_path else None
        self.conf_thresh = confidence_threshold
        self.iou_thresh = iou_threshold
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempts to load Ultralytics YOLO model."""
        if not self.weights_path or not self.weights_path.is_file():
            logger.info(
                "YOLO weights not provided or not found at '%s'. Using ContourDetector fallback.",
                self.weights_path,
            )
            return

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(str(self.weights_path))
            logger.info("Successfully loaded YOLO model from %s", self.weights_path)
        except Exception as e:
            logger.warning(
                "Failed to initialize Ultralytics YOLO from '%s' (%s). Falling back to ContourDetector.",
                self.weights_path,
                e,
            )
            self.model = None

    @property
    def is_available(self) -> bool:
        """Returns True if YOLO model is initialized and ready."""
        return self.model is not None

    def detect(self, image: np.ndarray) -> List[Component]:
        """Runs YOLO inference if loaded; otherwise invokes ContourDetector.

        Args:
            image: BGR image array.

        Returns:
            List of detected Component instances.
        """
        if not self.is_available:
            return ContourDetector().detect(image)

        try:
            results = self.model(
                image,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                verbose=False,
            )

            raw_components: List[Component] = []
            for r in results:
                boxes = r.boxes
                names = r.names
                for box in boxes:
                    cls_id = int(box.cls[0])
                    class_name = names.get(cls_id, f"class_{cls_id}").lower()
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    xmin, ymin, xmax, ymax = xyxy[:4]

                    raw_components.append(
                        Component(
                            id="",
                            class_name=class_name,
                            bbox=BoundingBox.from_xyxy(xmin, ymin, xmax, ymax),
                            confidence=conf,
                        )
                    )

            assigned = assign_reference_designators(raw_components)
            logger.info("YOLO detected %d components", len(assigned))
            return assigned

        except Exception as err:
            logger.error("YOLO inference failed: %s. Falling back to ContourDetector.", err)
            return ContourDetector().detect(image)


def detect_components(
    image: np.ndarray,
    weights_path: Optional[Union[str, Path]] = None,
    confidence_threshold: float = 0.25,
) -> List[Component]:
    """Convenience function to detect components in a PCB image.

    Args:
        image: BGR image array.
        weights_path: Optional path to YOLO .pt weights.
        confidence_threshold: Minimum detection confidence score.

    Returns:
        List of detected Component instances.
    """
    detector = YOLODetector(
        weights_path=weights_path,
        confidence_threshold=confidence_threshold,
    )
    return detector.detect(image)
