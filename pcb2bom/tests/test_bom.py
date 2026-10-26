"""Unit tests for the bom module."""

from pathlib import Path
import pandas as pd
import pytest

from pcb2bom.bom import (
    aggregate_bom,
    build_bom_entries,
    determine_bom_status,
    export_bom,
)
from pcb2bom.models import BoundingBox, Component, OCRCandidate, PartLookupResult


def test_determine_bom_status():
    # Identified IC
    assert determine_bom_status("ic", "mock", 0.95, 0.9) == "identified"

    # Low match confidence -> needs_review
    assert determine_bom_status("ic", "mock", 0.65, 0.85) == "needs_review"

    # Unidentified IC
    assert determine_bom_status("ic", "unidentified", 0.0, 0.8) == "unidentified"

    # Passive capacitor with good detection -> identified
    assert determine_bom_status("capacitor", "mock", 0.0, 0.75) == "identified"


def test_build_bom_entries_and_aggregation():
    # Create two identical capacitors
    c1 = Component(
        id="C1",
        class_name="capacitor",
        bbox=BoundingBox(10, 10, 25, 20),
        confidence=0.8,
        ocr_result=OCRCandidate("", "", 0.0, "0603 (estimated)", []),
        lookup_result=PartLookupResult("N/A", "0603-CAP", "Generic", "Ceramic Capacitor", "0603", "", 0.0, "passive"),
    )
    c2 = Component(
        id="C2",
        class_name="capacitor",
        bbox=BoundingBox(30, 10, 45, 20),
        confidence=0.84,
        ocr_result=OCRCandidate("", "", 0.0, "0603 (estimated)", []),
        lookup_result=PartLookupResult("N/A", "0603-CAP", "Generic", "Ceramic Capacitor", "0603", "", 0.0, "passive"),
    )
    # One distinct IC
    u1 = Component(
        id="U1",
        class_name="ic",
        bbox=BoundingBox(60, 60, 120, 100),
        confidence=0.92,
        ocr_result=OCRCandidate("NE555P", "NE555P", 0.9, "DIP-8 (estimated)", ["NE555P"]),
        lookup_result=PartLookupResult(
            "NE555P",
            "NE555P",
            "Texas Instruments",
            "Precision Timer",
            "DIP-8",
            "https://ti.com/ne555.pdf",
            1.0,
            "mock",
        ),
    )

    entries = build_bom_entries([c1, c2, u1])
    assert len(entries) == 3

    df_bom = aggregate_bom(entries)
    # Aggregated to 2 rows (1 for the grouped C1,C2, 1 for U1)
    assert len(df_bom) == 2

    # Verify capacitor aggregation
    cap_row = df_bom[df_bom["Component Class"] == "CAPACITOR"].iloc[0]
    assert "C1" in cap_row["Ref/ID"] and "C2" in cap_row["Ref/ID"]
    assert cap_row["Quantity"] == 2
    assert cap_row["Detection Confidence"] == pytest.approx(0.82, 0.01)

    # Verify IC row
    ic_row = df_bom[df_bom["Component Class"] == "IC"].iloc[0]
    assert ic_row["Ref/ID"] == "U1"
    assert ic_row["Part Number (OCR)"] == "NE555P"
    assert ic_row["Quantity"] == 1
    assert ic_row["Status"] == "identified"


def test_export_bom(tmp_path: Path):
    u1 = Component(
        id="U1",
        class_name="ic",
        bbox=BoundingBox(10, 10, 40, 40),
        confidence=0.9,
        ocr_result=OCRCandidate("LM358", "LM358", 0.9, "SOIC-8 (estimated)", []),
        lookup_result=PartLookupResult(
            "LM358", "LM358N", "Texas Instruments", "Op-Amp", "SOIC-8", "https://ti.com", 0.95, "mock"
        ),
    )

    csv_path, xlsx_path, df_bom = export_bom([u1], output_dir=tmp_path, base_name="test_bom")
    assert csv_path.exists()
    assert xlsx_path.exists()
    assert len(df_bom) == 1

    # Verify CSV content
    df_read = pd.read_csv(csv_path)
    assert len(df_read) == 1
    assert df_read.iloc[0]["Part Number (OCR)"] == "LM358N"
    assert df_read.iloc[0]["Manufacturer"] == "Texas Instruments"
