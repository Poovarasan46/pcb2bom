"""Utility script to generate sample boards and ground truth annotations for evaluation."""

from pathlib import Path
import json
import cv2
import numpy as np


def generate_sample_dataset(data_dir: Path) -> None:
    img_dir = data_dir / "sample_images"
    ann_dir = data_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Board 1: Timer Board -----------------
    b1 = np.full((600, 800, 3), (35, 115, 35), dtype=np.uint8)  # PCB Green
    # Traces
    cv2.line(b1, (50, 100), (750, 100), (45, 140, 200), 3)
    cv2.line(b1, (100, 50), (100, 550), (45, 140, 200), 3)
    cv2.line(b1, (300, 200), (300, 500), (45, 140, 200), 2)

    # IC1: NE555P (xmin=150, ymin=150, xmax=350, ymax=280)
    cv2.rectangle(b1, (150, 150), (350, 280), (25, 25, 25), -1)
    cv2.putText(b1, "NE555P", (180, 225), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2)

    # Passive 1: Capacitor C1 (xmin=420, ymin=160, xmax=470, ymax=210)
    cv2.rectangle(b1, (420, 160), (470, 210), (70, 100, 140), -1)

    # Passive 2: Resistor R1 (xmin=420, ymin=230, xmax=480, ymax=270)
    cv2.rectangle(b1, (420, 230), (480, 270), (15, 15, 15), -1)

    # Diode D1 (xmin=150, ymin=340, xmax=220, ymax=390)
    cv2.rectangle(b1, (150, 340), (220, 390), (10, 10, 10), -1)
    cv2.putText(b1, "1N4148", (155, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    # Transistor Q1 (xmin=270, ymin=340, xmax=340, ymax=400)
    cv2.rectangle(b1, (270, 340), (340, 400), (20, 20, 20), -1)
    cv2.putText(b1, "2N3904", (275, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1)

    cv2.imwrite(str(img_dir / "board_sample1.png"), b1)

    # ----------------- Board 2: Microcontroller Board -----------------
    b2 = np.full((600, 800, 3), (120, 45, 30), dtype=np.uint8)  # PCB Blue
    # Traces
    cv2.line(b2, (40, 120), (760, 120), (50, 160, 210), 3)
    cv2.line(b2, (120, 40), (120, 560), (50, 160, 210), 3)

    # IC1: STM32F401RET6 (xmin=220, ymin=180, xmax=460, ymax=360)
    cv2.rectangle(b2, (220, 180), (460, 360), (20, 20, 20), -1)
    cv2.putText(b2, "STM32F401", (245, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2)

    # IC2: AMS1117-3.3 (xmin=540, ymin=200, xmax=660, ymax=290)
    cv2.rectangle(b2, (540, 200), (660, 290), (30, 30, 30), -1)
    cv2.putText(b2, "AMS1117", (550, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)

    # Passive 1: Capacitor C1 (xmin=540, ymin=320, xmax=600, ymax=370)
    cv2.rectangle(b2, (540, 320), (600, 370), (75, 105, 145), -1)

    cv2.imwrite(str(img_dir / "board_sample2.png"), b2)

    # Ground truth annotations format
    annotations = [
        {
            "image": "board_sample1.png",
            "width": 800,
            "height": 600,
            "components": [
                {
                    "class": "ic",
                    "bbox": [150, 150, 350, 280],
                    "expected_ocr": "NE555P",
                    "expected_mpn": "NE555P",
                },
                {
                    "class": "capacitor",
                    "bbox": [420, 160, 470, 210],
                    "expected_ocr": "",
                    "expected_mpn": "",
                },
                {
                    "class": "resistor",
                    "bbox": [420, 230, 480, 270],
                    "expected_ocr": "",
                    "expected_mpn": "",
                },
                {
                    "class": "diode",
                    "bbox": [150, 340, 220, 390],
                    "expected_ocr": "1N4148",
                    "expected_mpn": "1N4148",
                },
                {
                    "class": "transistor",
                    "bbox": [270, 340, 340, 400],
                    "expected_ocr": "2N3904",
                    "expected_mpn": "2N3904",
                },
            ],
        },
        {
            "image": "board_sample2.png",
            "width": 800,
            "height": 600,
            "components": [
                {
                    "class": "ic",
                    "bbox": [220, 180, 460, 360],
                    "expected_ocr": "STM32F401",
                    "expected_mpn": "STM32F401RET6",
                },
                {
                    "class": "ic",
                    "bbox": [540, 200, 660, 290],
                    "expected_ocr": "AMS1117",
                    "expected_mpn": "AMS1117-3.3",
                },
                {
                    "class": "capacitor",
                    "bbox": [540, 320, 600, 370],
                    "expected_ocr": "",
                    "expected_mpn": "",
                },
            ],
        },
    ]

    ann_file = ann_dir / "ground_truth.json"
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    print(f"Sample dataset generated successfully in {data_dir}")


if __name__ == "__main__":
    generate_sample_dataset(Path(__file__).resolve().parent.parent / "data")
