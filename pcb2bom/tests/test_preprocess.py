"""Unit tests for the preprocess module."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from pcb2bom.preprocess import (
    apply_clahe,
    apply_gray_world,
    detect_glare_mask,
    fuse_multi_exposure,
    load_image,
    preprocess_image,
    reduce_glare_inpaint,
    resize_image,
    save_comparison_image,
)


def test_resize_image_downscales_large():
    large = np.zeros((3000, 4000, 3), dtype=np.uint8)
    resized, scale = resize_image(large, max_dim=2000)
    assert max(resized.shape[:2]) == 2000
    assert scale == 2000.0 / 4000.0
    assert resized.shape == (1500, 2000, 3)


def test_resize_image_no_op_for_small():
    small = np.zeros((300, 400, 3), dtype=np.uint8)
    resized, scale = resize_image(small, max_dim=1000)
    assert scale == 1.0
    assert resized.shape == (300, 400, 3)


def test_apply_clahe_preserves_shape_and_dtype(synthetic_pcb_image: np.ndarray):
    enhanced = apply_clahe(synthetic_pcb_image)
    assert enhanced.shape == synthetic_pcb_image.shape
    assert enhanced.dtype == np.uint8


def test_apply_gray_world(synthetic_pcb_image: np.ndarray):
    # Tint image heavily towards green
    tinted = synthetic_pcb_image.copy()
    tinted[:, :, 1] = np.clip(tinted[:, :, 1].astype(int) + 60, 0, 255).astype(np.uint8)

    balanced = apply_gray_world(tinted)
    assert balanced.shape == tinted.shape
    assert balanced.dtype == np.uint8


def test_detect_and_reduce_glare(synthetic_glare_image: np.ndarray):
    mask = detect_glare_mask(synthetic_glare_image, v_threshold=230, s_threshold=50)
    assert mask is not None
    assert np.count_nonzero(mask) > 0

    inpainted, returned_mask = reduce_glare_inpaint(synthetic_glare_image, mask)
    assert inpainted.shape == synthetic_glare_image.shape
    # Glare pixels in inpainted should no longer be pure white (255, 255, 255)
    glare_coords = np.where(mask > 0)
    if len(glare_coords[0]) > 0:
        sample_y, sample_x = glare_coords[0][0], glare_coords[1][0]
        # Inpainted value should blend into neighborhood
        assert not (inpainted[sample_y, sample_x] == [255, 255, 255]).all()


def test_fuse_multi_exposure():
    img1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    img2 = np.full((100, 100, 3), 150, dtype=np.uint8)
    fused = fuse_multi_exposure([img1, img2])
    assert fused.shape == (100, 100, 3)
    assert fused.dtype == np.uint8
    # Fused luminance should lie between dark and bright
    assert 50 < np.mean(fused) < 150


def test_fuse_multi_exposure_validation():
    with pytest.raises(ValueError):
        fuse_multi_exposure([np.zeros((10, 10, 3), dtype=np.uint8)])

    with pytest.raises(ValueError):
        fuse_multi_exposure([
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((20, 20, 3), dtype=np.uint8),
        ])


def test_save_comparison_image(tmp_path: Path, synthetic_pcb_image: np.ndarray):
    after = apply_clahe(synthetic_pcb_image)
    out_file = tmp_path / "comparison.jpg"
    saved = save_comparison_image(synthetic_pcb_image, after, out_file)
    assert saved.exists()
    assert saved.is_file()


def test_preprocess_image_pipeline(tmp_path: Path, synthetic_glare_image: np.ndarray):
    img_path = tmp_path / "test_board.png"
    cv2.imwrite(str(img_path), synthetic_glare_image)

    debug_path = tmp_path / "debug_comparison.png"
    res = preprocess_image(
        img_path,
        color_correction="clahe",
        fix_glare=True,
        max_dim=500,
        save_debug_to=debug_path,
    )
    assert res.processed is not None
    assert res.glare_mask is not None
    assert res.saved_debug_path is not None
    assert res.saved_debug_path.exists()
