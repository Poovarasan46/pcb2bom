"""Unit tests for the lookup module."""

from pathlib import Path
import pytest

from pcb2bom.lookup import (
    DiskCache,
    MockLookupProvider,
    PartLookupService,
    run_lookup_pipeline,
)
from pcb2bom.models import BoundingBox, Component, OCRCandidate


def test_disk_cache(tmp_path: Path):
    cache = DiskCache(cache_dir=tmp_path)
    assert cache.get("NE555P") is None

    sample_data = {"mpn": "NE555P", "manufacturer": "TI", "description": "Timer"}
    cache.set("NE555P", sample_data)

    retrieved = cache.get("NE555P")
    assert retrieved is not None
    assert retrieved["mpn"] == "NE555P"
    assert retrieved["manufacturer"] == "TI"


def test_mock_lookup_provider_exact():
    provider = MockLookupProvider()
    res = provider.search("NE555P")
    assert res is not None
    assert res["mpn"] == "NE555P"
    assert res["manufacturer"] == "Texas Instruments"
    assert res["score"] == 100.0


def test_mock_lookup_provider_fuzzy():
    provider = MockLookupProvider()
    # Test OCR confusion: NE55SP instead of NE555P
    res = provider.search("NE55SP")
    assert res is not None
    assert res["mpn"] == "NE555P"
    assert res["score"] >= 80.0

    # Substring / partial match
    res_mcu = provider.search("STM32F401")
    assert res_mcu is not None
    assert "STM32F401" in res_mcu["mpn"]


def test_mock_lookup_provider_unknown():
    provider = MockLookupProvider()
    res = provider.search("NONEXISTENT_PART_XYZ_99999")
    assert res is None


def test_part_lookup_service_mock_mode(tmp_path: Path):
    service = PartLookupService(use_mock=True, cache_dir=tmp_path)
    result = service.identify_part("LM358N", class_hint="ic")
    assert result.matched_mpn == "LM358N"
    assert result.manufacturer == "Texas Instruments"
    assert result.source == "mock"
    assert result.match_confidence > 0.8
    assert result.datasheet_url.startswith("http")


def test_part_lookup_service_unidentified(tmp_path: Path):
    service = PartLookupService(use_mock=True, cache_dir=tmp_path)
    result = service.identify_part("UNKNOWN_CHIP_123", class_hint="ic")
    assert result.source == "unidentified"
    assert result.match_confidence == 0.0


def test_part_lookup_service_empty_query(tmp_path: Path):
    service = PartLookupService(use_mock=True, cache_dir=tmp_path)
    result = service.identify_part("", class_hint="capacitor")
    assert result.source == "unidentified"
    assert result.match_confidence == 0.0


def test_run_lookup_pipeline(tmp_path: Path):
    c1 = Component(
        id="U1",
        class_name="ic",
        bbox=BoundingBox(10, 10, 50, 50),
        confidence=0.9,
        ocr_result=OCRCandidate(
            raw_text="NE555P",
            cleaned_text="NE555P",
            confidence=0.92,
            package_estimate="DIP-8 (estimated)",
            candidates=["NE555P"],
        ),
    )
    c2 = Component(
        id="U2",
        class_name="ic",
        bbox=BoundingBox(10, 80, 50, 120),
        confidence=0.85,
        ocr_result=OCRCandidate(
            raw_text="GARBAGE123",
            cleaned_text="GARBAGE123",
            confidence=0.4,
            package_estimate="SOIC-8 (estimated)",
            candidates=["GARBAGE123"],
        ),
    )
    c3 = Component(
        id="C1",
        class_name="capacitor",
        bbox=BoundingBox(70, 10, 90, 30),
        confidence=0.8,
        ocr_result=OCRCandidate(
            raw_text="",
            cleaned_text="",
            confidence=0.0,
            package_estimate="0603 (estimated)",
            candidates=[],
        ),
    )

    all_comps, unidentified = run_lookup_pipeline([c1, c2, c3], use_mock=True, cache_dir=tmp_path)
    assert len(all_comps) == 3
    assert c1.lookup_result.matched_mpn == "NE555P"
    assert len(unidentified) == 1
    assert unidentified[0].id == "U2"
