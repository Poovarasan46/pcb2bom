"""Unit tests for the detect module."""

import numpy as np
import pytest

from pcb2bom.detect import (
    ContourDetector,
    YOLODetector,
    apply_nms,
    assign_reference_designators,
    calculate_iou,
    detect_components,
)
from pcb2bom.models import BoundingBox, Component


def test_calculate_iou():
    box1 = BoundingBox(0, 0, 10, 10)
    box2 = BoundingBox(0, 0, 10, 10)
    assert calculate_iou(box1, box2) == 1.0

    box3 = BoundingBox(20, 20, 30, 30)
    assert calculate_iou(box1, box3) == 0.0

    # Half overlap: 5x10 intersection area = 50. Union = 100 + 100 - 50 = 150. IoU = 50/150 = 1/3
    box4 = BoundingBox(5, 0, 15, 10)
    assert pytest.approx(calculate_iou(box1, box4), 0.01) == 0.333


def test_apply_nms():
    comp1 = Component(
        id="",
        class_name="ic",
        bbox=BoundingBox(10, 10, 50, 50),
        confidence=0.9,
    )
    # Overlapping duplicate with lower confidence
    comp2 = Component(
        id="",
        class_name="ic",
        bbox=BoundingBox(12, 11, 51, 49),
        confidence=0.7,
    )
    # Distinct non-overlapping component
    comp3 = Component(
        id="",
        class_name="capacitor",
        bbox=BoundingBox(100, 100, 130, 130),
        confidence=0.85,
    )

    pruned = apply_nms([comp1, comp2, comp3], iou_threshold=0.4)
    assert len(pruned) == 2
    assert comp1 in pruned
    assert comp3 in pruned
    assert comp2 not in pruned


def test_assign_reference_designators():
    c1 = Component(id="", class_name="ic", bbox=BoundingBox(10, 10, 50, 50), confidence=0.9)
    c2 = Component(id="", class_name="ic", bbox=BoundingBox(10, 80, 50, 120), confidence=0.85)
    c3 = Component(id="", class_name="capacitor", bbox=BoundingBox(70, 10, 90, 30), confidence=0.8)
    c4 = Component(id="", class_name="resistor", bbox=BoundingBox(70, 40, 90, 60), confidence=0.8)

    assigned = assign_reference_designators([c1, c2, c3, c4])
    ids = [c.id for c in assigned]
    assert "U1" in ids
    assert "U2" in ids
    assert "C1" in ids
    assert "R1" in ids


def test_contour_detector(synthetic_pcb_image: np.ndarray):
    detector = ContourDetector(min_area=80)
    components = detector.detect(synthetic_pcb_image)

    assert len(components) >= 2
    # Verify valid bboxes and confidence scores
    for comp in components:
        assert comp.bbox.width > 0
        assert comp.bbox.height > 0
        assert 0.0 <= comp.confidence <= 1.0
        assert len(comp.id) >= 2


def test_yolo_detector_graceful_fallback(synthetic_pcb_image: np.ndarray):
    # Pass nonexistent weights file
    detector = YOLODetector(weights_path="nonexistent_yolo_weights.pt")
    assert not detector.is_available

    components = detector.detect(synthetic_pcb_image)
    assert isinstance(components, list)
    assert len(components) >= 2


def test_detect_components_api(synthetic_pcb_image: np.ndarray):
    components = detect_components(synthetic_pcb_image)
    assert isinstance(components, list)
    assert all(isinstance(c, Component) for c in components)
