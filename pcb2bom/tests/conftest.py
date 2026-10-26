"""Shared fixtures for pcb2bom tests."""

import numpy as np
import pytest
import cv2


@pytest.fixture
def synthetic_pcb_image() -> np.ndarray:
    """Creates a synthetic 400x400 PCB-like image with a green background and chip rectangles."""
    img = np.full((400, 400, 3), (34, 110, 34), dtype=np.uint8)  # PCB green (BGR)

    # Add copper traces (orange-gold lines)
    cv2.line(img, (20, 50), (380, 50), (45, 140, 200), 2)
    cv2.line(img, (50, 20), (50, 380), (45, 140, 200), 2)

    # IC 1 (Dark gray rectangle with white text)
    cv2.rectangle(img, (80, 80), (220, 200), (30, 30, 30), -1)
    cv2.putText(
        img,
        "NE555P",
        (100, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )

    # IC 2 (Another IC with marking)
    cv2.rectangle(img, (260, 100), (360, 220), (25, 25, 25), -1)
    cv2.putText(
        img,
        "LM358N",
        (275, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )

    # SMT capacitor (Small brownish rectangle)
    cv2.rectangle(img, (80, 260), (120, 280), (60, 90, 130), -1)

    # SMT resistor (Small black rectangle with white stripe)
    cv2.rectangle(img, (160, 260), (200, 280), (15, 15, 15), -1)

    return img


@pytest.fixture
def synthetic_glare_image(synthetic_pcb_image: np.ndarray) -> np.ndarray:
    """Adds specular reflection / glare spots to the synthetic image."""
    glare_img = synthetic_pcb_image.copy()
    # Saturated white specular highlight
    cv2.circle(glare_img, (150, 140), 20, (255, 255, 255), -1)
    cv2.circle(glare_img, (310, 160), 12, (252, 253, 255), -1)
    return glare_img
