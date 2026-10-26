"""Command-line interface for pcb2bom.

Usage:
    python -m pcb2bom run <image_path> [--out <dir>] [--mock] [--no-glare-fix] [--weights <path>]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from pcb2bom.pipeline import PipelineConfig, run_pipeline

logger = logging.getLogger("pcb2bom")


def setup_logging(verbose: bool = False) -> None:
    """Configures structured console logging."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "[%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=format_str, stream=sys.stdout)


def build_parser() -> argparse.ArgumentParser:
    """Constructs CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="pcb2bom",
        description="pcb2bom: Extract structured engineering Bills of Materials from PCB photos.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: run
    run_parser = subparsers.add_parser("run", help="Run detection and BoM generation on a PCB photo")
    run_parser.add_argument("image", type=str, help="Path to raw PCB image file")
    run_parser.add_argument(
        "--out",
        dest="out_dir",
        type=str,
        default="./output",
        help="Output directory for generated BoM files (default: ./output)",
    )
    run_parser.add_argument(
        "--mock",
        dest="use_mock",
        action="store_true",
        default=True,
        help="Use offline mock database for part lookups (default: True, no API keys needed)",
    )
    run_parser.add_argument(
        "--no-mock",
        dest="use_mock",
        action="store_false",
        help="Query live Digi-Key or Octopart APIs using environment credentials",
    )
    run_parser.add_argument(
        "--no-glare-fix",
        dest="fix_glare",
        action="store_false",
        default=True,
        help="Disable specular highlight / glare reduction filter",
    )
    run_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom YOLO fine-tuned weights (.pt file)",
    )
    run_parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold (default: 0.25)",
    )
    run_parser.add_argument(
        "--color-mode",
        choices=["clahe", "gray_world", "none"],
        default="clahe",
        help="Color normalization method (default: clahe)",
    )
    run_parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable detailed debug logging output",
    )

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Entrypoint function for CLI execution."""
    parser = build_parser()
    parsed = parser.parse_args(args)

    if not parsed.command:
        parser.print_help()
        return 0

    if parsed.command == "run":
        setup_logging(verbose=getattr(parsed, "verbose", False))

        image_path = Path(parsed.image)
        if not image_path.is_file():
            logger.error("Error: Specified input image does not exist: %s", image_path)
            return 1

        config = PipelineConfig(
            color_correction=parsed.color_mode,
            fix_glare=parsed.fix_glare,
            weights_path=parsed.weights,
            detection_confidence=parsed.conf,
            use_mock_lookup=parsed.use_mock,
            output_dir=Path(parsed.out_dir),
        )

        try:
            res = run_pipeline(image_path=image_path, config=config)
            print("\n================ pcb2bom Execution Summary ================")
            print(f"Components Detected : {len(res.components)}")
            print(f"BoM Line Items      : {len(res.bom_dataframe)}")
            print(f"Unidentified Parts  : {len(res.unidentified_components)}")
            print(f"CSV BoM Saved To   : {res.csv_path}")
            print(f"Excel BoM Saved To : {res.xlsx_path}")
            print(f"Annotated Image     : {res.annotated_image_path}")
            print("===========================================================\n")
            return 0
        except Exception as e:
            logger.exception("Pipeline run encountered a fatal error: %s", e)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
