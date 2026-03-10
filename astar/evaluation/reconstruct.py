"""
Report Reconstruction Module

Reconstruct free-text reports from structured data.
This is used for evaluation of information fidelity.

Input: structured_output.json (structured data)
Output: reconstructed_output.json (reconstructed descriptions)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from astar.core.config import get_config, load_config
from astar.core.llm import call_llm
from astar.core.utils import load_json, save_json

# System prompt for report reconstruction (Chinese medical domain)
RECONSTRUCT_SYSTEM = """You are a medical report writing expert. Based on the given structured data, reconstruct the original description text.

Structured data:
{structured}

Please generate a complete description paragraph based on the structured data above.
Requirements:
- Use standard medical report language style
- Include all information from the structured fields
- Do not add information not present in the structured data

Output the description text directly, without JSON or other formats."""


def reconstruct_single(structured: Dict) -> Dict:
    """
    Reconstruct description from structured data.

    Args:
        structured: Structured extraction dictionary.

    Returns:
        Dictionary with reconstructed text and success flag.
    """
    if not structured:
        return {"reconstructed_desc": "", "success": False}

    system = RECONSTRUCT_SYSTEM.format(
        structured=json.dumps(structured, ensure_ascii=False, indent=2)
    )

    result = call_llm(system, "Please reconstruct the description text.")

    if result:
        return {
            "reconstructed_desc": result.strip(),
            "success": True,
        }
    return {"reconstructed_desc": "", "success": False}


def reconstruct_reports(
    structured_path: str,
    output_path: Optional[str] = None,
    max_workers: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Batch reconstruct reports from structured data.

    Args:
        structured_path: Path to structured_output.json.
        output_path: Output path for reconstructed reports.
        max_workers: Number of parallel workers.
        config_path: Path to config file.

    Returns:
        Dictionary with reconstruction results.
    """
    # Load config
    if config_path:
        load_config(config_path)
    cfg = get_config()

    if max_workers is None:
        max_workers = cfg.evaluation.max_workers

    # Load structured data
    structured_data = load_json(Path(structured_path))
    results_list = structured_data.get("results", [])
    print(f"[Reconstruct] Loaded {len(results_list)} structured reports")

    # Prepare items (only non-empty extractions)
    items = []
    for item in results_list:
        rec_id = str(item.get("id", ""))
        extraction = item.get("extraction", {})
        if extraction:
            items.append((rec_id, extraction))

    print(f"[Reconstruct] Processing {len(items)} reports")

    # Reconstruct in parallel
    results = []
    success_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(reconstruct_single, item[1]): item[0]
            for item in items
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reconstructing"):
            rec_id = futures[fut]
            try:
                result = fut.result()
                result["id"] = rec_id
                results.append(result)
                if result["success"]:
                    success_count += 1
            except Exception as e:
                results.append({
                    "id": rec_id,
                    "reconstructed_desc": "",
                    "success": False,
                    "error": str(e),
                })

    print(f"[Reconstruct] Done. Success: {success_count}/{len(results)}")

    # Build output
    output = {
        "source": str(structured_path),
        "num_reports": len(results),
        "success_count": success_count,
        "results": results,
    }

    # Save results
    if output_path is None:
        output_path = Path(structured_path).parent / "reconstructed_output.json"
    save_json(output, Path(output_path))

    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Reconstruct reports from structured data",
    )
    parser.add_argument(
        "--structured",
        required=True,
        help="Path to structured output JSON",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file",
    )
    args = parser.parse_args()

    reconstruct_reports(
        args.structured,
        args.output,
        args.workers,
        args.config,
    )


if __name__ == "__main__":
    main()
