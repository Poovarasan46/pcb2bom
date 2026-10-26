"""OCR module for reading component markings from PCB crops.

Includes crop enhancement (upscaling, unsharp masking, binarization),
text sanitization and OCR confusion normalization (0/O, 1/I, 5/S),
multi-engine OCR execution (EasyOCR, pytesseract, fallback),
and dimensional package type estimation heuristics.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pcb2bom.models import BoundingBox, Component, OCRCandidate

logger = logging.getLogger(__name__)


def crop_component(
    image: np.ndarray,
    bbox: BoundingBox,
    margin_px: int = 4,
) -> np.ndarray:
    """Extracts a component sub-image bounded within image dimensions.

    Args:
        image: Full BGR image array.
        bbox: Component BoundingBox.
        margin_px: Padding margin in pixels.

    Returns:
        Cropped BGR image array.
    """
    h, w = image.shape[:2]
    ymin = max(0, bbox.ymin - margin_px)
    ymax = min(h, bbox.ymax + margin_px)
    xmin = max(0, bbox.xmin - margin_px)
    xmax = min(w, bbox.xmax + margin_px)

    if ymax <= ymin or xmax <= xmin:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    return image[ymin:ymax, xmin:xmax].copy()


def enhance_crop_for_ocr(
    crop: np.ndarray,
    upscale_factor: float = 2.0,
    apply_sharpen: bool = True,
) -> np.ndarray:
    """Prepares a component crop for OCR extraction via upscaling and sharpening.

    Args:
        crop: BGR crop array.
        upscale_factor: Scaling factor to enlarge small laser-etched markings.
        apply_sharpen: Whether to apply unsharp masking.

    Returns:
        Enhanced grayscale image ready for OCR.
    """
    if crop.size == 0:
        return crop

    # 1. Convert to grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop.copy()

    # 2. Upscale small chips so character strokes have sufficient pixel width
    if upscale_factor > 1.0:
        new_w = int(round(gray.shape[1] * upscale_factor))
        new_h = int(round(gray.shape[0] * upscale_factor))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # 3. Unsharp masking to enhance laser etching edges
    if apply_sharpen:
        gaussian = cv2.GaussianBlur(gray, (0, 0), 2.0)
        sharpened = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    else:
        sharpened = gray

    # 4. Normalize contrast
    enhanced = cv2.normalize(sharpened, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return enhanced


def clean_ocr_text(raw_text: str) -> str:
    """Cleans and sanitizes raw OCR output into a standardized electronic part string.

    - Strips non-alphanumeric noise (preserves valid MPN hyphens and slashes)
    - Removes whitespace and lowercase clutter
    - Filters common OCR punctuation artifacts

    Args:
        raw_text: Unprocessed OCR text.

    Returns:
        Cleaned uppercase string.
    """
    if not raw_text:
        return ""

    # Replace newlines and separators with spaces
    text = raw_text.replace("\n", " ").replace("\r", " ")

    # Strip symbols that are almost never in MPNs
    cleaned = re.sub(r"[^A-Za-z0-9\-_/]", "", text)
    cleaned = cleaned.strip("-_/").upper()
    return cleaned


def generate_candidate_permutations(text: str) -> List[str]:
    """Generates candidate part numbers by resolving common OCR optical confusions.

    Standard optical confusion pairs in laser-etched PCB packages:
    - '0' (zero) <-> 'O' (letter)
    - '1' (one)  <-> 'I' (letter)
    - '5' (five) <-> 'S' (letter)
    - '8' (eight)<-> 'B' (letter)

    Args:
        text: Sanitized part string.

    Returns:
        List of candidate strings with permutations.
    """
    if not text:
        return []

    candidates = {text}

    # Substitution mapping
    confusions = [
        ("0", "O"),
        ("O", "0"),
        ("1", "I"),
        ("I", "1"),
        ("5", "S"),
        ("S", "5"),
        ("8", "B"),
        ("B", "8"),
    ]

    for char_from, char_to in confusions:
        if char_from in text:
            perm = text.replace(char_from, char_to)
            candidates.add(perm)

    return sorted(list(candidates))


def estimate_package(
    class_name: str,
    bbox: BoundingBox,
) -> str:
    """Heuristically estimates package type from bounding box dimensions and class.

    Args:
        class_name: Component classification ('ic', 'capacitor', etc.).
        bbox: Component BoundingBox.

    Returns:
        Package description string labelled as '(estimated)'.
    """
    cls = class_name.lower()
    w, h = bbox.width, bbox.height
    long_side = max(w, h)
    short_side = min(w, h)
    aspect = float(long_side / short_side) if short_side > 0 else 1.0

    if cls == "ic":
        if aspect <= 1.25:
            if long_side < 40:
                return "QFN / DFN (estimated)"
            if long_side < 100:
                return "TQFP / QFP (estimated)"
            return "BGA / QFP (estimated)"
        elif 1.25 < aspect <= 3.5:
            if long_side < 50:
                return "SOIC / MSOP (estimated)"
            return "SOIC / SOP (estimated)"
        else:
            return "DIP / SIP (estimated)"

    elif cls in ("capacitor", "resistor", "inductor"):
        if long_side < 15:
            return "0402 / 0603 (estimated)"
        elif long_side < 28:
            return "0805 / 1206 (estimated)"
        elif long_side < 50:
            return "1210 / 2512 (estimated)"
        return "Through-Hole / Radial (estimated)"

    elif cls == "diode":
        if aspect >= 2.0:
            return "SMA / SMB / SOD-123 (estimated)"
        return "SOT-23 / DO-214 (estimated)"

    elif cls == "transistor":
        if aspect < 1.4:
            return "SOT-23 (estimated)"
        return "SOT-223 / TO-252 (estimated)"

    elif cls == "connector":
        return "Header / Terminal (estimated)"

    elif cls == "crystal":
        return "SMD Crystal / HC-49 (estimated)"

    return "SMD Generic (estimated)"


class OCREngine:
    """Orchestrates OCR extraction across available backends (EasyOCR -> pytesseract -> synthetic)."""

    def __init__(self) -> None:
        self._easyocr_reader = None
        self._easyocr_attempted = False
        self._pytesseract_available: Optional[bool] = None

    def _init_easyocr(self) -> None:
        """Attempts lazy initialization of EasyOCR."""
        if self._easyocr_attempted:
            return
        self._easyocr_attempted = True
        try:
            import easyocr  # type: ignore

            # Initialize for English numbers/alphanumerics
            self._easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            logger.info("Initialized EasyOCR engine successfully.")
        except Exception as e:
            logger.info("EasyOCR not available (%s); falling back to secondary OCR.", e)
            self._easyocr_reader = None

    def _check_pytesseract(self) -> bool:
        """Checks if pytesseract is available and configured."""
        if self._pytesseract_available is not None:
            return self._pytesseract_available
        try:
            import pytesseract  # type: ignore

            # Probe whether tesseract is installed in system path
            pytesseract.get_tesseract_version()
            self._pytesseract_available = True
            logger.info("Initialized pytesseract engine successfully.")
        except Exception:
            self._pytesseract_available = False
            logger.info("pytesseract not available or executable not in PATH.")
        return self._pytesseract_available

    def extract_text(self, crop: np.ndarray) -> Tuple[str, float]:
        """Runs OCR on an enhanced component crop.

        Args:
            crop: Enhanced image array.

        Returns:
            Tuple of (raw_text, confidence_score).
        """
        if crop.size == 0:
            return "", 0.0

        # Try EasyOCR first
        self._init_easyocr()
        if self._easyocr_reader is not None:
            try:
                results = self._easyocr_reader.readtext(crop)
                if results:
                    # Sort results by confidence and length
                    best = max(results, key=lambda item: (len(item[1]), item[2]))
                    raw_text = best[1]
                    conf = float(best[2])
                    return raw_text, conf
            except Exception as e:
                logger.debug("EasyOCR run failed: %s", e)

        # Fallback to pytesseract
        if self._check_pytesseract():
            try:
                import pytesseract  # type: ignore

                # Use OEM 3 (Default) and PSM 6 (Assume a single uniform block of text)
                data = pytesseract.image_to_data(
                    crop, output_type=pytesseract.Output.DICT, config="--psm 6"
                )
                texts = [
                    data["text"][i]
                    for i in range(len(data["text"]))
                    if int(data["conf"][i]) > 0 and data["text"][i].strip()
                ]
                confs = [
                    float(data["conf"][i]) / 100.0
                    for i in range(len(data["text"]))
                    if int(data["conf"][i]) > 0 and data["text"][i].strip()
                ]
                if texts:
                    joined = " ".join(texts)
                    mean_conf = sum(confs) / len(confs)
                    return joined, mean_conf
            except Exception as e:
                logger.debug("pytesseract run failed: %s", e)

        return "", 0.0


# Shared singleton instance
_GLOBAL_OCR_ENGINE: Optional[OCREngine] = None


def get_ocr_engine() -> OCREngine:
    global _GLOBAL_OCR_ENGINE
    if _GLOBAL_OCR_ENGINE is None:
        _GLOBAL_OCR_ENGINE = OCREngine()
    return _GLOBAL_OCR_ENGINE


def process_component_ocr(
    image: np.ndarray,
    component: Component,
    engine: Optional[OCREngine] = None,
) -> OCRCandidate:
    """Executes the full OCR and packaging heuristic pipeline for a single component.

    Args:
        image: Full board image.
        component: Target Component instance.
        engine: Optional pre-configured OCREngine.

    Returns:
        OCRCandidate instance.
    """
    pkg_estimate = estimate_package(component.class_name, component.bbox)

    # For passive resistors/capacitors without laser etched part numbers,
    # skip heavy OCR to save latency and avoid reading PCB silk noise
    if component.class_name.lower() in ("resistor", "capacitor"):
        candidate = OCRCandidate(
            raw_text="",
            cleaned_text="",
            confidence=0.0,
            package_estimate=pkg_estimate,
            candidates=[],
        )
        component.ocr_result = candidate
        return candidate

    crop = crop_component(image, component.bbox)
    enhanced = enhance_crop_for_ocr(crop)

    ocr_eng = engine or get_ocr_engine()
    raw_text, conf = ocr_eng.extract_text(enhanced)
    cleaned = clean_ocr_text(raw_text)
    perms = generate_candidate_permutations(cleaned)

    candidate = OCRCandidate(
        raw_text=raw_text,
        cleaned_text=cleaned,
        confidence=conf,
        package_estimate=pkg_estimate,
        candidates=perms,
    )
    component.ocr_result = candidate
    return candidate


def run_ocr_pipeline(
    image: np.ndarray,
    components: List[Component],
) -> List[Component]:
    """Applies OCR and package estimation across all components.

    Args:
        image: Full PCB image.
        components: List of detected components.

    Returns:
        Components updated with OCRCandidate results.
    """
    engine = get_ocr_engine()
    for comp in components:
        process_component_ocr(image, comp, engine=engine)
    logger.info("Completed OCR processing on %d components", len(components))
    return components
