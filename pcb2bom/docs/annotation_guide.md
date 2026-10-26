# PCB Component Dataset Annotation Guide

This guide establishes the annotation protocol for training and fine-tuning YOLO component detectors for **pcb2bom**. Following these standards ensures consistent bounding boxes, high detection accuracy, and seamless dataset integration.

---

## 1. Class Taxonomy & Designator Mappings

We define **8 core component classes** covering over 95% of discrete and active devices on standard surface-mount (SMD) and through-hole (THT) printed circuit boards:

| Class ID | Class Name | Reference Prefix | Description & Common Package Styles |
| :--- | :--- | :--- | :--- |
| **0** | `ic` | `U` | Microcontrollers, op-amps, logic, flash memory (DIP, SOIC, TSSOP, QFP, QFN, BGA). |
| **1** | `capacitor` | `C` | Multilayer ceramic caps (MLCC), electrolytic cans, tantalum (0402, 0603, 0805, 1206, Radial). |
| **2** | `resistor` | `R` | Thick/thin film chip resistors, resistor arrays, carbon film THT (0402, 0603, 0805, 1206). |
| **3** | `connector` | `J` | Header pins, USB-C/Micro, screw terminals, RJ45, HDMI, barrel jacks. |
| **4** | `inductor` | `L` | Power inductors, ferrite beads, wirewound chokes (SMD drum, toroidal, 0805 beads). |
| **5** | `crystal` | `Y` | Quartz crystals, ceramic resonators, TCXOs (HC-49S, SMD 3225, SMD 5032). |
| **6** | `diode` | `D` | Rectifier diodes, Zeners, Schottky, LEDs (SOD-123, SMA, SMB, SMC, 0805 LED). |
| **7** | `transistor` | `Q` | BJTs, MOSFETs, LDO voltage regulators (SOT-23, SOT-223, DPAK, TO-220, TO-92). |

---

## 2. Bounding Box Conventions

### 2.1 Tight Enclosing Box
- Draw bounding boxes as tightly as possible around the **physical component body and exposed package leads**.
- Do **not** include PCB silkscreen reference labels (e.g. `R12`, `C4`) inside the component bounding box.
- For SMD passives, include the exposed metallic end-caps (solder terminations).

### 2.2 Handling Overlaps and High-Density Clusters
- **Decoupling clusters:** Decoupling capacitors often cluster right next to IC power pins. Ensure each individual capacitor has its own distinct bounding box.
- **Occluded components:** If a small component is partially shadowed by a tall heat sink or tall connector, annotate only if >= 60% of the body is visible.

### 2.3 Connectors and Header Pins
- For male pin headers (e.g., 2.54mm 1x6 header), draw a **single enclosing box** around the entire connector assembly rather than individual pin boxes.

---

## 3. YOLO Format Specification

Annotations must follow standard YOLO format with normalized floating-point coordinates relative to the image width and height:

```
<class_id> <x_center> <y_center> <width> <height>
```

- `class_id`: Integer index from `0` to `7` according to the taxonomy table above.
- `x_center`, `y_center`: Center point of bounding box, normalized to `[0.0, 1.0]`.
- `width`, `height`: Box dimensions, normalized to `[0.0, 1.0]`.

### Coordinate Conversion Formula
Given image width $W$, image height $H$, and pixel box $[x_{min}, y_{min}, x_{max}, y_{max}]$:
$$\text{x\_center} = \frac{x_{min} + x_{max}}{2 \cdot W}, \quad \text{y\_center} = \frac{y_{min} + y_{max}}{2 \cdot H}$$
$$\text{width} = \frac{x_{max} - x_{min}}{W}, \quad \text{height} = \frac{y_{max} - y_{min}}{H}$$

---

## 4. Directory Structure for Training

Store the dataset in standard Ultralytics YOLO structure:

```
dataset/
├── dataset.yaml
├── images/
│   ├── train/
│   │   ├── board_001.jpg
│   │   └── board_002.jpg
│   └── val/
│       └── board_003.jpg
└── labels/
    ├── train/
    │   ├── board_001.txt
    │   └── board_002.txt
    └── val/
        └── board_003.txt
```

### `dataset.yaml` Configuration File:
```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: ic
  1: capacitor
  2: resistor
  3: connector
  4: inductor
  5: crystal
  6: diode
  7: transistor
```

---

## 5. Tooling Recommendations & Export

1. **LabelImg / CVAT / Roboflow**:
   - When using **LabelImg**: Set format to **YOLO** in the left sidebar and load `classes.txt` with the 8 classes above.
   - When using **Roboflow**: Select **YOLOv8 PyTorch** export format.
2. **Cropping Helper**:
   - Use `python scripts/crop_and_label_helper.py -i ./raw_photos -o ./labeled_crops` to automatically segment candidate components into categorized folders for manual validation.
