"""Unit and integration tests for the pipeline and CLI."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from pcb2bom.cli import build_parser, main
from pcb2bom.models import BoundingBox, Component, OCRCandidate, PartLookupResult
from pcb2bom.pipeline import (
    PipelineConfig,
    draw_component_annotations,
    run_pipeline,
)


def test_draw_component_annotations(tmp_path: Path, synthetic_pcb_image: np.ndarray):
    u1 = Component(
        id="U1",
        class_name="ic",
        bbox=BoundingBox(50, 50, 150, 150),
        confidence=0.95,
        ocr_result=OCRCandidate("NE555P", "NE555P", 0.9, "DIP-8 (estimated)", []),
        lookup_result=PartLookupResult("NE555P", "NE555P", "TI", "Timer", "DIP-8", "", 1.0, "mock"),
    )
    c1 = Component(
        id="C1",
        class_name="capacitor",
        bbox=BoundingBox(180, 50, 200, 70),
        confidence=0.8,
        ocr_result=None,
        lookup_result=None,
    )

    out_file = tmp_path / "annotated.jpg"
    saved = draw_component_annotations(synthetic_pcb_image, [u1, c1], out_file)
    assert saved.exists()
    assert saved.is_file()

    # Verify image is valid
    read_back = cv2.imread(str(saved))
    assert read_back is not None
    assert read_back.shape == synthetic_pcb_image.shape


def test_run_pipeline_end_to_end(tmp_path: Path, synthetic_pcb_image: np.ndarray):
    img_file = tmp_path / "board_input.png"
    cv2.imwrite(str(img_file), synthetic_pcb_image)

    out_dir = tmp_path / "pipeline_output"
    config = PipelineConfig(
        color_correction="clahe",
        fix_glare=True,
        max_image_dim=800,
        use_mock_lookup=True,
        cache_dir=tmp_path / "cache",
        output_dir=out_dir,
    )

    result = run_pipeline(image_path=img_file, config=config)

    assert result.csv_path.exists()
    assert result.xlsx_path.exists()
    assert result.annotated_image_path.exists()
    assert len(result.components) >= 2
    assert len(result.bom_dataframe) >= 1
    assert "Ref/ID" in result.bom_dataframe.columns
    assert "Component Class" in result.bom_dataframe.columns


def test_cli_parser_and_run(tmp_path: Path, synthetic_pcb_image: np.ndarray):
    img_file = tmp_path / "cli_board.jpg"
    cv2.imwrite(str(img_file), synthetic_pcb_image)

    out_dir = tmp_path / "cli_output"
    args = [
        "run",
        str(img_file),
        "--out",
        str(out_dir),
        "--mock",
        "--color-mode",
        "clahe",
    ]

    exit_code = main(args)
    assert exit_code == 0
    assert (out_dir / "bom.csv").exists()
    assert (out_dir / "bom.xlsx").exists()
    assert (out_dir / "annotated_board.jpg").exists()
