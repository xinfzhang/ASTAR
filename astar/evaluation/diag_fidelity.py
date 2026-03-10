"""
Diagnostic Fidelity Evaluation

Evaluate whether structured data retains sufficient information to support diagnosis:
1. Read structured_output.json
2. Infer diagnosis
3. Compare with GT CSV ground truth diagnosis
4. Compute: PDA (Primary Diagnosis Accuracy), KFP (Key Finding Preservation), CA (Clinical Actionability)
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
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

# System prompt for diagnosis inference
INFER_DIAGNOSIS_SYSTEM = """You are a radiologist. Infer the diagnosis based on the following structured examination data.

Structured data:
{structured}

Output JSON (output JSON only):
{{"primary_diagnosis": "main diagnosis", "key_findings": ["finding1", "finding2"], "recommendations": "recommendations"}}"""

# System prompt for LLM-as-Judge evaluation
JUDGE_SYSTEM = """You are a medical report quality evaluation expert. Compare the inferred diagnosis with the ground truth diagnosis and score.

Inferred diagnosis:
{inferred}

Ground truth diagnosis:
{ground_truth}

Output JSON (output JSON only, no other content):
{{"pda": 0-5 integer, "kfp": 0-1 decimal, "ca": 0-5 integer, "reason": "scoring rationale"}}"""


def infer_diagnosis(structured: Dict) -> Dict:
    """
    Infer diagnosis from structured data.

    Args:
        structured: Structured extraction dictionary.

    Returns:
        Inferred diagnosis dictionary.
    """
    if not structured:
        return {}

    system = INFER_DIAGNOSIS_SYSTEM.format(
        structured=json.dumps(structured, ensure_ascii=False, indent=2)
    )
    result = call_llm_json(system, "Please infer the diagnosis, output JSON only")
    return result if result else {}


def judge_diagnosis(inferred: Dict, ground_truth: str) -> Dict:
    """
    Judge the consistency between inferred and ground truth diagnosis.

    Args:
        inferred: Inferred diagnosis dictionary.
        ground_truth: Ground truth diagnosis string.

    Returns:
        Scoring dictionary with pda, kfp, ca metrics.
    """
    if not inferred:
        return {"pda": 0.0, "kfp": 0.0, "ca": 0.0}

    system = JUDGE_SYSTEM.format(
        inferred=json.dumps(inferred, ensure_ascii=False, indent=2),
        ground_truth=ground_truth,
    )

    result = call_llm_json(system, "Please evaluate, output JSON only")
    if result:
        pda = result.get("pda", 0)
        kfp = result.get("kfp", 0)
        ca = result.get("ca", 0)

        # Normalize to 0-1 range
        if isinstance(pda, (int, float)):
            pda = min(5, max(0, pda)) / 5.0
        else:
            pda = 0.0

        if isinstance(kfp, (int, float)):
            kfp = min(1.0, max(0.0, float(kfp)))
        else:
            kfp = 0.0

        if isinstance(ca, (int, float)):
            ca = min(5, max(0, ca)) / 5.0
        else:
            ca = 0.0

        return {
            "pda": pda,
            "kfp": kfp,
            "ca": ca,
            "reason": result.get("reason", ""),
        }

    return {"pda": 0.0, "kfp": 0.0, "ca": 0.0}


def evaluate_single(structured: Dict, gt_diag: str) -> Optional[Dict]:
    """
    Evaluate a single report.

    Args:
        structured: Structured extraction dictionary.
        gt_diag: Ground truth diagnosis.

    Returns:
        Evaluation result dictionary.
    """
    # Infer diagnosis
    inferred = infer_diagnosis(structured)
    if not inferred:
        return None

    # Judge
    scores = judge_diagnosis(inferred, gt_diag)

    return {
        "inferred_diagnosis": inferred,
        "scores": scores,
    }


def evaluate_diag_fidelity(
    structured_path: str,
    gt_path: str,
    output_path: str = None,
    max_workers: int = None,
    config_path: str = None,
) -> Dict:
    """
    Evaluate diagnostic fidelity metrics.

    Args:
        structured_path: Path to structured_output.json.
        gt_path: Path to GT CSV file.
        output_path: Output path for results.
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

    # Load structured results
    structured_data = load_json(Path(structured_path))
    results_list = structured_data.get("results", [])
    print(f"[DiagFidelity] Loaded {len(results_list)} structured reports")

    # Load GT data
    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    gt_df.columns = gt_df.columns.str.strip()

    # Build ID to GT diagnosis mapping
    id_column = cfg.data.id_column
    id_to_diag = {}
    for idx, row in gt_df.iterrows():
        diag = str(row.get(cfg.data.diagnosis_column, "") or "")

        if id_column in gt_df.columns:
            rec_id = str(row[id_column]).strip()
            id_to_diag[rec_id] = diag

        idx_id = str(idx)
        id_to_diag[idx_id] = diag

    # Prepare evaluation items
    eval_items = []
    for item in results_list:
        rec_id = str(item.get("id", ""))
        structured = item.get("extraction", {})
        if not structured:
            continue
        gt_diag = id_to_diag.get(rec_id, "")
        eval_items.append((rec_id, structured, gt_diag))

    print(f"[DiagFidelity] Evaluating {len(eval_items)} reports")

    if not eval_items:
        print("No valid items to evaluate")
        return {"avg_pda": 0, "avg_kfp": 0, "avg_ca": 0, "num_reports": 0}

    # Evaluate in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_single, item[1], item[2]): item[0]
            for item in eval_items
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            try:
                result = fut.result()
                if result:
                    result["id"] = futures[fut]
                    results.append(result)
            except Exception as e:
                print(f"Error: {e}")

    # Calculate averages
    n = len(results)
    if n == 0:
        print("No successful results")
        return {"avg_pda": 0, "avg_kfp": 0, "avg_ca": 0, "num_reports": 0}

    avg_pda = sum(r["scores"]["pda"] for r in results) / n
    avg_kfp = sum(r["scores"]["kfp"] for r in results) / n
    avg_ca = sum(r["scores"]["ca"] for r in results) / n

    print(f"\n{'=' * 50}")
    print("Diagnostic Fidelity Evaluation Results")
    print(f"{'=' * 50}")
    print(f"PDA (Primary Diagnosis Accuracy): {avg_pda:.4f}")
    print(f"KFP (Key Finding Preservation): {avg_kfp:.4f}")
    print(f"CA (Clinical Actionability): {avg_ca:.4f}")

    # Build output
    output = {
        "avg_pda": avg_pda,
        "avg_kfp": avg_kfp,
        "avg_ca": avg_ca,
        "num_reports": n,
        "details": results,
    }

    # Save results
    if output_path is None:
        output_path = Path(structured_path).parent / "eval_diag_fidelity.json"
    save_json(output, Path(output_path))

    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate diagnostic fidelity (PDA, KFP, CA)",
    )
    parser.add_argument(
        "--structured",
        required=True,
        help="Path to structured_output.json",
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

    evaluate_diag_fidelity(
        args.structured,
        args.gt,
        args.output,
        args.workers,
        args.config,
    )


if __name__ == "__main__":
    main()
