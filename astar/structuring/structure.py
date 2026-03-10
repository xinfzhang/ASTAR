"""
Report Structuring Module

This module provides functionality to structure medical reports
using an induced template.

Usage:
    from astar.structuring.structure import structure_reports
    results = structure_reports("data/reports.csv", "output/template_final.json")
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from astar.core.config import get_config, load_config
from astar.core.llm import call_llm_json
from astar.core.utils import load_json, save_json

from astar.structuring.template_utils import (
    flatten_template,
    slots_to_simple_template,
    unflatten_result,
    extract_filled_values,
)

# System prompt for report structuring (Chinese medical domain)
STRUCTURE_REPORT_SYSTEM = """You are a medical report structuring expert. Extract structured information from the given medical report according to the template.

Template fields:
{template}

Report content:
{report}

Instructions:
- Extract values for each field based on the report content
- Use null for fields not mentioned in the report
- For categorical fields, use values from the allowed_values list
- For numerical fields, include units if present

Output JSON format:
{output_format}"""


def structure_single_report(
    report: Dict[str, str],
    report_id: str,
    slots: List[Dict],
    simple_template: Dict[str, str],
    key_to_path: Dict[str, str],
) -> Dict[str, Any]:
    """
    Structure a single report using the template.

    Args:
        report: Report dictionary with description and diagnosis.
        report_id: Unique identifier for the report.
        slots: List of template slots.
        simple_template: Simplified template for prompts.
        key_to_path: Key to path mapping.

    Returns:
        Structured extraction result.
    """
    cfg = get_config()

    desc = report.get(cfg.data.description_column, "") or ""
    diag = report.get(cfg.data.diagnosis_column, "") or ""

    if not desc and not diag:
        return {
            "id": report_id,
            "extraction": {},
            "success": False,
            "error": "Empty report",
        }

    # Build prompt
    template_str = json.dumps(simple_template, ensure_ascii=False, indent=2)
    report_str = f"Description:\n{desc}\n\nDiagnosis:\n{diag}"

    # Expected output format
    output_format = {k: None for k in simple_template.keys()}
    output_format_str = json.dumps(output_format, ensure_ascii=False, indent=2)

    system = STRUCTURE_REPORT_SYSTEM.format(
        template=template_str,
        report=report_str,
        output_format=output_format_str,
    )

    result = call_llm_json(system, "Please extract structured information.")

    if result and isinstance(result, dict):
        return {
            "id": report_id,
            "extraction": result,
            "success": True,
        }
    else:
        return {
            "id": report_id,
            "extraction": {},
            "success": False,
            "error": "LLM extraction failed",
        }


def structure_reports(
    data_path: str,
    template_path: str,
    output_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    num_reports: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Structure multiple reports using a template.

    Args:
        data_path: Path to CSV file with reports.
        template_path: Path to template JSON file.
        output_path: Output path for structured results.
        max_workers: Number of parallel workers.
        num_reports: Maximum reports to process.
        config_path: Path to config file.

    Returns:
        Dictionary with structured results.
    """
    # Load config
    if config_path:
        load_config(config_path)
    cfg = get_config()

    if max_workers is None:
        max_workers = cfg.structuring.max_workers

    # Load template
    template = load_json(Path(template_path))
    slots, key_to_path = flatten_template(template)
    simple_template = slots_to_simple_template(slots)
    print(f"[Structure] Loaded template with {len(slots)} slots")

    # Load data
    df = pd.read_csv(data_path, encoding="utf-8")
    df.columns = df.columns.str.strip()

    if num_reports:
        df = df.head(num_reports)
    print(f"[Structure] Processing {len(df)} reports")

    # Determine ID column
    id_column = cfg.data.id_column
    if id_column not in df.columns:
        # Fall back to index
        df["_id"] = df.index.astype(str)
        id_column = "_id"

    # Prepare work items
    items = [
        (row.to_dict(), str(row.get(id_column, idx)))
        for idx, row in df.iterrows()
    ]

    # Process in parallel
    results = []
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                structure_single_report,
                item[0],
                item[1],
                slots,
                simple_template,
                key_to_path,
            ): item[1]
            for item in items
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Structuring"):
            report_id = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                if result.get("success"):
                    success_count += 1
            except Exception as e:
                results.append({
                    "id": report_id,
                    "extraction": {},
                    "success": False,
                    "error": str(e),
                })

    print(f"[Structure] Done. Success: {success_count}/{len(results)}")

    # Build output
    output = {
        "template_path": str(template_path),
        "data_path": str(data_path),
        "num_reports": len(results),
        "success_count": success_count,
        "results": results,
    }

    # Save results
    if output_path is None:
        output_path = Path(data_path).parent / "structured_output.json"
    save_json(output, Path(output_path))

    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Structure medical reports using an induced template",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV file with reports",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template JSON file (template_final.json)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for structured results",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--num_reports",
        type=int,
        default=None,
        help="Maximum reports to process",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file",
    )
    args = parser.parse_args()

    structure_reports(
        data_path=args.data,
        template_path=args.template,
        output_path=args.output,
        max_workers=args.workers,
        num_reports=args.num_reports,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
