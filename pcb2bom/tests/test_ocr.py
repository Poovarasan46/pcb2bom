"""Unit tests for the ocr module."""

import numpy as np
import pytest

from pcb2bom.models import BoundingBox, Component
from pcb2bom.ocr import (
    OCREngine,
    clean_ocr_text,
    crop_component,
    enhance_crop_for_ocr,
    estimate_package,
    generate_candidate_permutations,
    process_component_ocr,
    run_ocr_pipeline,
)


def test_crop_component():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:40, 20:40] = 255

    bbox = BoundingBox(20, 20, 40, 40)
    crop = crop_component(img, bbox, margin_px=5)
    # y: 15 to 45 (30px), x: 15 to 45 (30px)
    assert crop.shape == (30, 30, 3)

    # Edge clamping at (0, 0)
    bbox_edge = BoundingBox(0, 0, 10, 10)
    crop_edge = crop_component(img, bbox_edge, margin_px=5)
    assert crop_edge.shape == (15, 15, 3)


def test_enhance_crop_for_ocr():
    crop = np.full((30, 40, 3), 120, dtype=np.uint8)
    enhanced = enhance_crop_for_ocr(crop, upscale_factor=2.0)
    assert enhanced.shape == (60, 80)
    assert enhanced.dtype == np.uint8


def test_clean_ocr_text():
    assert clean_ocr_text("  NE555P  \n") == "NE555P"
    assert clean_ocr_text("STM32-F401/RE ~.") == "STM32-F401/RE"
    assert clean_ocr_text(" [MAX232] ") == "MAX232"
    assert clean_ocr_text("") == ""


def test_generate_candidate_permutations():
    cands = generate_candidate_permutations("NE555P")
    assert "NE555P" in cands
    assert "NE55SP" in cands or "NE5SSP" in cands or "NESSSP" in cands

    cands_o0 = generate_candidate_permutations("MAX01")
    assert "MAX01" in cands_o0
    assert "MAXO1" in cands_o0 or "MAXOI" in cands_o0


def test_estimate_package():
    # Square IC -> QFN or QFP
    square_ic = BoundingBox(10, 10, 40, 40)
    est = estimate_package("ic", square_ic)
    assert "QFN" in est or "QFP" in est
    assert "(estimated)" in est

    # Elongated IC -> SOIC / SOP
    rect_ic = BoundingBox(10, 10, 60, 30)
    est_rect = estimate_package("ic", rect_ic)
    assert "SOIC" in est_rect or "SOP" in est_rect

    # Passives
    tiny_res = BoundingBox(0, 0, 10, 5)
    assert "0402" in estimate_package("resistor", tiny_res)

    med_cap = BoundingBox(0, 0, 25, 12)
    assert "0805" in estimate_package("capacitor", med_cap)


def test_ocr_engine_fallback():
    engine = OCREngine()
    dummy = np.zeros((20, 20), dtype=np.uint8)
    text, conf = engine.extract_text(dummy)
    assert isinstance(text, str)
    assert isinstance(conf, float)


def test_process_component_ocr():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    comp = Component(
        id="U1",
        class_name="ic",
        bbox=BoundingBox(50, 50, 120, 90),
        confidence=0.9,
    )
    cand = process_component_ocr(img, comp)
    assert comp.ocr_result is not None
    assert comp.ocr_result == cand
    assert "(estimated)" in cand.package_estimate


def test_run_ocr_pipeline():
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    comp1 = Component(id="U1", class_name="ic", bbox=BoundingBox(10, 10, 60, 40), confidence=0.9)
    comp2 = Component(id="C1", class_name="capacitor", bbox=BoundingBox(80, 80, 90, 85), confidence=0.8)

    comps = run_ocr_pipeline(img, [comp1, comp2])
    assert comps[0].ocr_result is not None
    assert comps[1].ocr_result is not None
