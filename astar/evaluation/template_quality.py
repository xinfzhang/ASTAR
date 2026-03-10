"""
Template Quality Evaluation

Evaluate how completely the template covers clinical concepts in unseen reports:
- Cov_case: Case-level coverage (macro average)
- Cov_key: Key-level coverage (micro average)

Input: template_final.json + GT CSV
"""

import random
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional
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
from astar.structuring.template_utils import extract_template_keys

# System prompt for extracting concept keys from reports
EXTRACT_KEYS_SYSTEM = """You are a medical report structuring expert. Extract all clinical concept keys from the given medical report.

Each concept key is in "entity+attribute" form, e.g., fetal_biparietal_diameter, lateral_ventricle_dilation, placenta_position

Output JSON (output JSON only):
{"keys": ["concept_key_1", "concept_key_2", ...]}"""

# System prompt for semantic matching
MATCH_KEY_SYSTEM = """Determine if the report concept key semantically matches any template key.

Report concept key: {report_key}
Template key set: {template_keys}

Output JSON (output JSON only):
{{"match": true or false, "matched_key": "matched template key"}}"""


def extract_report_keys(report: Dict) -> List[str]:
    """
    Extract concept keys from a report.

    Args:
        report: Report dictionary.

    Returns:
        List of concept keys.
    """
    cfg = get_config()
    desc = str(report.get(cfg.data.description_column, "") or "")
    diag = str(report.get(cfg.data.diagnosis_column, "") or "")

    user_prompt = f"Report content:\n{desc}\n{diag}"
    result = call_llm_json(EXTRACT_KEYS_SYSTEM, user_prompt)

    if result and "keys" in result:
        keys = result["keys"]
        if isinstance(keys, list):
            return [str(k) for k in keys if k]
    return []


def check_key_match(report_key: str, template_keys: Set[str]) -> bool:
    """
    Check if a report key matches any template key.

    Args:
        report_key: Key extracted from report.
        template_keys: Set of template keys.

    Returns:
        True if match found.
    """
    # Simple string matching first
    report_key_lower = report_key.lower()
    for tk in template_keys:
        tk_lower = tk.lower()
        if report_key_lower in tk_lower or tk_lower in report_key_lower:
            return True

    # LLM semantic matching
    template_keys_sample = random.sample(
        list(template_keys), min(30, len(template_keys))
    )
    system = MATCH_KEY_SYSTEM.format(
        report_key=report_key,
        template_keys=", ".join(template_keys_sample),
    )

    result = call_llm_json(system, "Determine if there is a match")
    if result:
        match = result.get("match")
        if isinstance(match, bool):
            return match
        if isinstance(match, str):
            return match.lower() in ["true", "yes"]
    return False


def evaluate_single_report(
    report: Dict,
    template_keys: Set[str],
) -> Optional[Dict]:
    """
    Evaluate coverage for a single report.

    Args:
        report: Report dictionary.
        template_keys: Set of template keys.

    Returns:
        Coverage metrics for this report.
    """
    report_keys = extract_report_keys(report)
    if not report_keys:
        return None

    matched = 0
    for key in report_keys:
        if check_key_match(key, template_keys):
            matched += 1

    return {
        "total_keys": len(report_keys),
        "matched_keys": matched,
        "coverage": matched / len(report_keys),
        "report_keys": report_keys[:10],
    }


def evaluate_template_quality(
    template_path: str,
    gt_path: str,
    output_path: Optional[str] = None,
    num_reports: Optional[int] = None,
    max_workers: Optional[int] = None,
    config_path: Optional[str] = None,
) -> Dict:
    """
    Evaluate template quality metrics.

    Args:
        template_path: Path to template_final.json.
        gt_path: Path to GT CSV file.
        output_path: Output path for results.
        num_reports: Number of reports to evaluate.
        max_workers: Number of parallel workers.
        config_path: Path to config file.

    Returns:
        Dictionary with evaluation results.
    """
    # Load config
    if config_path:
        load_config(config_path)
    cfg = get_config()

    if max_workers is None:
        max_workers = cfg.evaluation.max_workers
    if num_reports is None:
        num_reports = cfg.evaluation.num_reports

    # Load template
    template = load_json(Path(template_path))
    template_keys = extract_template_keys(template)
    print(f"[TemplateQuality] Template has {len(template_keys)} keys")

    # Load GT reports
    df = pd.read_csv(gt_path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    num_reports = min(num_reports, len(df))
    indices = random.sample(range(len(df)), num_reports)
    reports = [df.iloc[i].to_dict() for i in indices]
    print(f"[TemplateQuality] Evaluating {num_reports} reports")

    # Evaluate in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single_report, r, template_keys): i
            for i, r in enumerate(reports)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = fut.result()
                if result:
                    results.append(result)
            except Exception as e:
                print(f"Error: {e}")

    # Calculate metrics
    if not results:
        print("No successful results")
        return {"cov_case": 0, "cov_key": 0, "num_reports": 0}

    total_keys_all = sum(r["total_keys"] for r in results)
    matched_keys_all = sum(r["matched_keys"] for r in results)

    cov_case = sum(r["coverage"] for r in results) / len(results)
    cov_key = matched_keys_all / total_keys_all if total_keys_all > 0 else 0

    print(f"\n{'=' * 50}")
    print("Template Quality Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Cov_case (case-level coverage): {cov_case:.4f}")
    print(f"Cov_key (key-level coverage): {cov_key:.4f}")
    print(f"Total concept keys: {total_keys_all}")
    print(f"Matched keys: {matched_keys_all}")

    # Build output
    output = {
        "cov_case": cov_case,
        "cov_key": cov_key,
        "total_keys": total_keys_all,
        "matched_keys": matched_keys_all,
        "num_reports": len(results),
        "details": results,
    }

    # Save results
    if output_path is None:
        output_path = Path(template_path).parent / "eval_template_quality.json"
    save_json(output, Path(output_path))

    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate template quality (coverage metrics)",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template_final.json",
    )
    parser.add_argument(
        "--gt",
        required=True,
        help="Path to GT CSV file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path",
    )
    parser.add_argument(
        "--num_reports",
        type=int,
        default=None,
        help="Number of reports to evaluate",
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

    evaluate_template_quality(
        args.template,
        args.gt,
        args.output,
        args.num_reports,
        args.workers,
        args.config,
    )


if __name__ == "__main__":
    main()
