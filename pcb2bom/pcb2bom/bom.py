"""Bill of Materials (BoM) structuring and export module.

Groups detected and identified components, aggregates quantities,
computes confidence metrics, and exports clean, formatted CSV and Excel files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd

from pcb2bom.models import BoMEntry, Component

logger = logging.getLogger(__name__)

BOM_COLUMNS: List[str] = [
    "Ref/ID",
    "Component Class",
    "Part Number (OCR)",
    "Manufacturer",
    "Description",
    "Package",
    "Quantity",
    "Detection Confidence",
    "Match Confidence",
    "Datasheet URL",
    "Status",
]


def determine_bom_status(
    class_name: str,
    match_source: str,
    match_conf: float,
    detect_conf: float,
) -> str:
    """Calculates status: 'identified', 'unidentified', or 'needs_review'.

    Args:
        class_name: Component class ('ic', 'capacitor', etc.).
        match_source: Lookup source ('digikey', 'mock', 'unidentified', etc.).
        match_conf: Fuzzy match confidence (0.0 to 1.0).
        detect_conf: Object detection confidence (0.0 to 1.0).

    Returns:
        Status string.
    """
    cls_lower = class_name.lower()

    # Passives (resistors, capacitors) often have no laser markings
    if cls_lower in ("resistor", "capacitor", "inductor"):
        if detect_conf >= 0.6:
            return "identified"
        return "needs_review"

    # ICs / Discretes
    if match_source == "unidentified" or match_conf <= 0.0:
        return "unidentified"

    if match_conf >= 0.80 and detect_conf >= 0.50:
        return "identified"

    return "needs_review"


def build_bom_entries(components: List[Component]) -> List[BoMEntry]:
    """Builds individual raw BoM entries for each component.

    Args:
        components: List of processed Component instances.

    Returns:
        List of un-aggregated BoMEntry instances.
    """
    entries: List[BoMEntry] = []

    for comp in components:
        ocr = comp.ocr_result
        lookup = comp.lookup_result

        raw_ocr = ocr.cleaned_text if ocr and ocr.cleaned_text else "N/A"
        matched_mpn = lookup.matched_mpn if lookup and lookup.matched_mpn != "N/A" else raw_ocr
        mfg = lookup.manufacturer if lookup else "Unknown"
        desc = lookup.description if lookup else ""
        pkg = (
            lookup.package
            if (lookup and lookup.package not in ("Unknown", "SMD/Through-Hole"))
            else (ocr.package_estimate if ocr else "Unknown")
        )
        url = lookup.datasheet_url if lookup else ""
        m_conf = lookup.match_confidence if lookup else 0.0
        src = lookup.source if lookup else "unidentified"

        status = determine_bom_status(
            class_name=comp.class_name,
            match_source=src,
            match_conf=m_conf,
            detect_conf=comp.confidence,
        )

        entries.append(
            BoMEntry(
                ref_id=comp.id,
                component_class=comp.class_name.upper(),
                part_number=matched_mpn,
                manufacturer=mfg,
                description=desc,
                package=pkg,
                quantity=1,
                detection_confidence=comp.confidence,
                match_confidence=m_conf,
                datasheet_url=url,
                status=status,
            )
        )

    return entries


def aggregate_bom(entries: List[BoMEntry]) -> pd.DataFrame:
    """Aggregates identical parts by grouping part number, class, package, and status.

    Concatenates reference designators (e.g. 'C1, C2, C3'), sums quantity,
    and averages confidence scores.

    Args:
        entries: Raw list of individual BoMEntry objects.

    Returns:
        Clean, sorted pandas DataFrame matching standard engineering BoM format.
    """
    if not entries:
        return pd.DataFrame(columns=BOM_COLUMNS)

    df_raw = pd.DataFrame([e.to_dict() for e in entries])

    # Grouping keys
    group_cols = [
        "Component Class",
        "Part Number (OCR)",
        "Manufacturer",
        "Description",
        "Package",
        "Datasheet URL",
        "Status",
    ]

    grouped = (
        df_raw.groupby(group_cols, as_index=False, dropna=False)
        .agg({
            "Ref/ID": lambda refs: ", ".join(sorted(refs, key=lambda r: (r[:1], int(r[1:]) if r[1:].isdigit() else 999))),
            "Quantity": "sum",
            "Detection Confidence": "mean",
            "Match Confidence": "mean",
        })
    )

    # Reorder columns to standard format
    df_bom = grouped[BOM_COLUMNS].copy()

    # Sort logically: ICs first, then passives, then connectors
    def class_sort_order(cls_name: str) -> int:
        order = {"IC": 1, "TRANSISTOR": 2, "DIODE": 3, "CRYSTAL": 4, "INDUCTOR": 5, "CAPACITOR": 6, "RESISTOR": 7, "CONNECTOR": 8}
        return order.get(str(cls_name).upper(), 99)

    df_bom["_sort_key"] = df_bom["Component Class"].apply(class_sort_order)
    df_bom = df_bom.sort_values(by=["_sort_key", "Part Number (OCR)"]).drop(columns=["_sort_key"]).reset_index(drop=True)

    # Format numeric confidence columns
    df_bom["Detection Confidence"] = df_bom["Detection Confidence"].round(3)
    df_bom["Match Confidence"] = df_bom["Match Confidence"].round(3)

    return df_bom


def export_bom(
    components: List[Component],
    output_dir: Union[str, Path],
    base_name: str = "bom",
) -> Tuple[Path, Path, pd.DataFrame]:
    """Generates BoM and exports both CSV and Excel (.xlsx) files.

    Args:
        components: Processed components.
        output_dir: Destination directory.
        base_name: File base name (default: 'bom').

    Returns:
        Tuple of (csv_path, xlsx_path, df_bom).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    entries = build_bom_entries(components)
    df_bom = aggregate_bom(entries)

    csv_path = out_path / f"{base_name}.csv"
    xlsx_path = out_path / f"{base_name}.xlsx"

    # 1. Export CSV
    df_bom.to_csv(csv_path, index=False, encoding="utf-8")
    logger.info("Exported CSV BoM (%d rows) to %s", len(df_bom), csv_path)

    # 2. Export Excel with column styling
    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df_bom.to_excel(writer, sheet_name="Bill of Materials", index=False)
        logger.info("Exported Excel BoM to %s", xlsx_path)
    except Exception as e:
        logger.warning("Excel export failed (%s); CSV was saved successfully.", e)

    return csv_path, xlsx_path, df_bom
