"""Image preprocessing module for the pcb2bom pipeline.

Provides color normalization (CLAHE / Gray-World), specular reflection
detection and inpainting, multi-exposure fusion, and side-by-side artifact saving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Output container for image preprocessing."""

    original: np.ndarray
    processed: np.ndarray
    glare_mask: Optional[np.ndarray]
    scale_factor: float
    saved_debug_path: Optional[Path] = None


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """Loads an image from disk using OpenCV.

    Args:
        image_path: Path to the image file.

    Returns:
        BGR image array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be decoded as an image.
    """
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at path: {path}")

    # Use cv2.imdecode to safely handle Unicode/Windows paths
    with open(path, "rb") as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Failed to decode image from path: {path}")

    logger.debug("Loaded image '%s' with shape %s", path.name, image.shape)
    return image


def resize_image(image: np.ndarray, max_dim: int = 2048) -> Tuple[np.ndarray, float]:
    """Resizes an image preserving aspect ratio if larger than max_dim.

    Args:
        image: BGR image array.
        max_dim: Maximum width or height.

    Returns:
        Tuple of (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim or max_dim <= 0:
        return image.copy(), 1.0

    scale = max_dim / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized image from (%d, %d) to (%d, %d) (scale=%.4f)", w, h, new_w, new_h, scale)
    return resized, scale


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Applies Contrast Limited Adaptive Histogram Equalization on LAB L-channel.

    Args:
        image: BGR image array.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Size of grid for histogram equalization.

    Returns:
        Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_equalized = clahe.apply(l_channel)

    merged_lab = cv2.merge((l_equalized, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    logger.debug("Applied CLAHE (clip_limit=%.1f, tile=%s)", clip_limit, tile_grid_size)
    return enhanced_bgr


def apply_gray_world(image: np.ndarray) -> np.ndarray:
    """Applies the Gray-World white balance algorithm.

    Scales each color channel based on the assumption that the average scene
    reflectance is achromatic (gray).

    Args:
        image: BGR image array.

    Returns:
        White-balanced BGR image.
    """
    b, g, r = cv2.split(image.astype(np.float32))
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)

    gray_target = (b_mean + g_mean + r_mean) / 3.0

    # Avoid division by zero
    b_scale = gray_target / (b_mean + 1e-6)
    g_scale = gray_target / (g_mean + 1e-6)
    r_scale = gray_target / (r_mean + 1e-6)

    b_corrected = np.clip(b * b_scale, 0, 255)
    g_corrected = np.clip(g * g_scale, 0, 255)
    r_corrected = np.clip(r * r_scale, 0, 255)

    corrected = cv2.merge((b_corrected, g_corrected, r_corrected)).astype(np.uint8)
    logger.debug("Applied Gray-World correction (b=%.2f, g=%.2f, r=%.2f)", b_scale, g_scale, r_scale)
    return corrected


def detect_glare_mask(
    image: np.ndarray,
    v_threshold: int = 238,
    s_threshold: int = 50,
    morph_kernel_size: int = 5,
) -> np.ndarray:
    """Detects specular highlights (glare) using HSV color space thresholding.

    Specular highlights typically exhibit very high brightness (Value)
    and low color saturation (Saturation).

    Args:
        image: BGR image array.
        v_threshold: Value channel brightness threshold (0-255).
        s_threshold: Saturation channel ceiling threshold (0-255).
        morph_kernel_size: Morphological dilation kernel size.

    Returns:
        Binary mask (uint8, 0 or 255) where 255 indicates glare.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    _, s_chan, v_chan = cv2.split(hsv)

    # Glare condition: high brightness and low saturation
    glare_mask = ((v_chan >= v_threshold) & (s_chan <= s_threshold)).astype(np.uint8) * 255

    # Filter isolated noisy pixels and dilate slightly to cover boundary halos
    if morph_kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel)
        glare_mask = cv2.dilate(glare_mask, kernel, iterations=1)

    glare_pixels = int(np.count_nonzero(glare_mask))
    total_pixels = image.shape[0] * image.shape[1]
    pct = (glare_pixels / total_pixels) * 100.0
    logger.debug("Glare detected: %d pixels (%.2f%% of image)", glare_pixels, pct)
    return glare_mask


def reduce_glare_inpaint(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    inpaint_radius: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Inpaints detected specular highlights using Telea's algorithm.

    Args:
        image: BGR image array.
        mask: Optional binary mask. If None, detected automatically.
        inpaint_radius: Radius for neighborhood sampling during inpainting.

    Returns:
        Tuple of (inpainted_image, glare_mask).
    """
    if mask is None:
        mask = detect_glare_mask(image)

    if np.count_nonzero(mask) == 0:
        logger.debug("No glare detected; skipping inpainting.")
        return image.copy(), mask

    inpainted = cv2.inpaint(image, mask, inpaintRadius=inpaint_radius, flags=cv2.INPAINT_TELEA)
    logger.debug("Applied inpainting to glare regions (radius=%d)", inpaint_radius)
    return inpainted, mask


def fuse_multi_exposure(images: List[np.ndarray]) -> np.ndarray:
    """Fuses multiple photos with different exposures using Mertens exposure fusion.

    Args:
        images: List of BGR images (must be registered / same dimensions).

    Returns:
        Fused BGR image (uint8).

    Raises:
        ValueError: If less than 2 images are provided or dimensions mismatch.
    """
    if len(images) < 2:
        raise ValueError("Multi-exposure fusion requires at least 2 images.")

    base_shape = images[0].shape
    for idx, img in enumerate(images[1:], start=1):
        if img.shape != base_shape:
            raise ValueError(
                f"Image at index {idx} has shape {img.shape}, expected {base_shape}."
            )

    merge_mertens = cv2.createMergeMertens()
    # OpenCV MergeMertens accepts standard uint8 BGR images and returns float32 in [0, 1]
    uint8_images = [img if img.dtype == np.uint8 else np.clip(img, 0, 255).astype(np.uint8) for img in images]
    fused_32f = merge_mertens.process(uint8_images)
    fused_8u = np.clip(fused_32f * 255.0, 0, 255).astype(np.uint8)
    logger.info("Fused %d multi-exposure images successfully", len(images))
    return fused_8u


def save_comparison_image(
    before: np.ndarray,
    after: np.ndarray,
    output_path: Union[str, Path],
    label_before: str = "Original",
    label_after: str = "Preprocessed",
) -> Path:
    """Concatenates before and after images horizontally with text labels for documentation.

    Args:
        before: Original BGR image.
        after: Processed BGR image.
        output_path: Destination file path.
        label_before: Text label for original.
        label_after: Text label for processed.

    Returns:
        Path to the saved image.
    """
    h1, w1 = before.shape[:2]
    h2, w2 = after.shape[:2]
    target_h = max(h1, h2)

    # Resize to equal height
    b_resized = cv2.resize(before, (int(w1 * (target_h / h1)), target_h))
    a_resized = cv2.resize(after, (int(w2 * (target_h / h2)), target_h))

    # Add text labels with a dark banner
    def add_label(img: np.ndarray, text: str) -> np.ndarray:
        res = img.copy()
        cv2.rectangle(res, (0, 0), (res.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(
            res,
            text,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return res

    b_labeled = add_label(b_resized, label_before)
    a_labeled = add_label(a_resized, label_after)

    comparison = np.hstack([b_labeled, a_labeled])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), comparison)
    logger.info("Saved before/after comparison to: %s", out)
    return out


def preprocess_image(
    input_source: Union[str, Path, np.ndarray],
    color_correction: str = "clahe",
    fix_glare: bool = True,
    max_dim: int = 2048,
    save_debug_to: Optional[Union[str, Path]] = None,
) -> PreprocessResult:
    """Executes the complete preprocessing pipeline on a single image.

    Args:
        input_source: Path to image file or existing numpy BGR array.
        color_correction: 'clahe', 'gray_world', or 'none'.
        fix_glare: Whether to detect and inpaint specular reflections.
        max_dim: Max image dimension for resizing (0 to disable).
        save_debug_to: Optional path to save before/after comparison.

    Returns:
        PreprocessResult containing original, processed image, glare mask, and scale.
    """
    if isinstance(input_source, (str, Path)):
        original = load_image(input_source)
    elif isinstance(input_source, np.ndarray):
        original = input_source.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(input_source)}")

    # 1. Resize if required
    resized, scale = resize_image(original, max_dim=max_dim)
    working = resized.copy()

    # 2. Glare reduction
    glare_mask = None
    if fix_glare:
        working, glare_mask = reduce_glare_inpaint(working)

    # 3. Color correction
    if color_correction.lower() == "clahe":
        working = apply_clahe(working)
    elif color_correction.lower() in ("gray_world", "grayworld"):
        working = apply_gray_world(working)
    elif color_correction.lower() == "none":
        pass
    else:
        logger.warning("Unknown color correction '%s'; falling back to CLAHE", color_correction)
        working = apply_clahe(working)

    # 4. Optional debug comparison image
    debug_path = None
    if save_debug_to is not None:
        debug_path = save_comparison_image(resized, working, save_debug_to)

    return PreprocessResult(
        original=original,
        processed=working,
        glare_mask=glare_mask,
        scale_factor=scale,
        saved_debug_path=debug_path,
    )
