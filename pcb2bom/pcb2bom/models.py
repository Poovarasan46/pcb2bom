"""Data models and data structures for the pcb2bom pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class BoundingBox:
    """Represents an axis-aligned bounding box (xmin, ymin, xmax, ymax)."""

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    def __post_init__(self) -> None:
        if self.xmin > self.xmax:
            raise ValueError(f"xmin ({self.xmin}) cannot be greater than xmax ({self.xmax})")
        if self.ymin > self.ymax:
            raise ValueError(f"ymin ({self.ymin}) cannot be greater than ymax ({self.ymax})")

    @property
    def width(self) -> int:
        return max(0, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(0, self.ymax - self.ymin)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        """Returns width / height. Defaults to 1.0 if height is 0."""
        return float(self.width / self.height) if self.height > 0 else 1.0

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    @classmethod
    def from_xyxy(cls, xmin: int, ymin: int, xmax: int, ymax: int) -> BoundingBox:
        return cls(int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax)))

    @classmethod
    def from_xywh(cls, x: int, y: int, w: int, h: int) -> BoundingBox:
        return cls(int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h)))


@dataclass
class Component:
    """Detected PCB component representation."""

    id: str
    class_name: str
    bbox: BoundingBox
    confidence: float
    ocr_result: Optional[OCRCandidate] = None
    lookup_result: Optional[PartLookupResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "class_name": self.class_name,
            "bbox": self.bbox.as_tuple(),
            "confidence": round(self.confidence, 4),
            "ocr": self.ocr_result.to_dict() if self.ocr_result else None,
            "lookup": self.lookup_result.to_dict() if self.lookup_result else None,
        }


@dataclass
class OCRCandidate:
    """Results from the OCR stage for a component."""

    raw_text: str
    cleaned_text: str
    confidence: float
    package_estimate: str
    candidates: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "confidence": round(self.confidence, 4),
            "package_estimate": self.package_estimate,
            "candidates": self.candidates,
        }


@dataclass
class PartLookupResult:
    """Component identification and catalog lookup result."""

    query_string: str
    matched_mpn: str
    manufacturer: str
    description: str
    package: str
    datasheet_url: str
    match_confidence: float
    source: str = "mock"  # "digikey", "octopart", "mock", "unidentified"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_string": self.query_string,
            "matched_mpn": self.matched_mpn,
            "manufacturer": self.manufacturer,
            "description": self.description,
            "package": self.package,
            "datasheet_url": self.datasheet_url,
            "match_confidence": round(self.match_confidence, 4),
            "source": self.source,
        }


@dataclass
class BoMEntry:
    """Row representation for the generated Bill of Materials."""

    ref_id: str
    component_class: str
    part_number: str
    manufacturer: str
    description: str
    package: str
    quantity: int
    detection_confidence: float
    match_confidence: float
    datasheet_url: str
    status: str  # "identified", "unidentified", "needs_review"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Ref/ID": self.ref_id,
            "Component Class": self.component_class,
            "Part Number (OCR)": self.part_number,
            "Manufacturer": self.manufacturer,
            "Description": self.description,
            "Package": self.package,
            "Quantity": self.quantity,
            "Detection Confidence": round(self.detection_confidence, 4),
            "Match Confidence": round(self.match_confidence, 4),
            "Datasheet URL": self.datasheet_url,
            "Status": self.status,
        }
