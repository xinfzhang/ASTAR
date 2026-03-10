"""
Information Fidelity Evaluation (ROUGE)

Evaluate whether structured data can completely restore the original report
(description section only).

Input:
  - reconstructed_output.json (reconstructed descriptions)
  - gt.csv (original reports)
Output:
  - ROUGE-1/2/L scores
"""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

try:
    import jieba
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False
    print("Warning: rouge_score not installed, run: pip install rouge_score jieba")

from astar.core.config import get_config, load_config
from astar.core.utils import load_json, save_json


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE scores.

    Args:
        reference: Reference text.
        hypothesis: Hypothesis text.

    Returns:
        Dictionary with rouge1, rouge2, rougeL F1 scores.
    """
    if not HAS_ROUGE:
        print("Warning: rouge_score not installed, cannot compute ROUGE")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    if not reference or not hypothesis:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    # Chinese word segmentation
    ref_tok = " ".join(jieba.cut(reference))
    hyp_tok = " ".join(jieba.cut(hypothesis))

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(ref_tok, hyp_tok)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


def evaluate_info_fidelity(
    reconstructed_path: str,
    gt_path: str,
    output_path: str = None,
    config_path: str = None,
) -> Dict:
    """
    Evaluate information fidelity using ROUGE.

    Args:
        reconstructed_path: Path to reconstructed_output.json.
        gt_path: Path to GT CSV file.
        output_path: Output path for results.
        config_path: Path to config file.

    Returns:
        Dictionary with evaluation results.
    """
    # Load config
    if config_path:
        load_config(config_path)
    cfg = get_config()

    # Load reconstructed reports
    reconstructed_data = load_json(Path(reconstructed_path))
    results_list = reconstructed_data.get("results", [])
    print(f"[InfoFidelity] Loaded {len(results_list)} reconstructed reports")

    success_count = sum(1 for r in results_list if r.get("success", False))
    print(f"[InfoFidelity] Successfully reconstructed: {success_count}")

    # Load GT data
    gt_df = pd.read_csv(gt_path, encoding="utf-8")
    gt_df.columns = gt_df.columns.str.strip()

    # Build ID to GT description mapping
    id_column = cfg.data.id_column
    desc_column = cfg.data.description_column

    id_to_gt_desc = {}
    for idx, row in gt_df.iterrows():
        if id_column in gt_df.columns:
            rec_id = str(row[id_column]).strip()
            id_to_gt_desc[rec_id] = str(row.get(desc_column, "") or "")
        # Also use row index as backup ID
        idx_id = str(idx)
        id_to_gt_desc[idx_id] = str(row.get(desc_column, "") or "")

    print(f"[InfoFidelity] GT reports: {len(gt_df)}")

    # Compute ROUGE
    results = []
    for item in tqdm(results_list, desc="Computing ROUGE"):
        rec_id = str(item.get("id", ""))

        # Skip failed reconstructions
        if not item.get("success", False):
            continue

        reconstructed_desc = item.get("reconstructed_desc", "")
        gt_desc = id_to_gt_desc.get(rec_id, "")

        if not gt_desc:
            print(f"Warning: ID {rec_id} not found in GT")
            continue

        rouge_scores = compute_rouge(gt_desc, reconstructed_desc)

        results.append({
            "id": rec_id,
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "gt_length": len(gt_desc),
            "reconstructed_length": len(reconstructed_desc),
        })

    # Calculate averages
    n = len(results)
    if n == 0:
        print("No successful results to evaluate")
        return {"avg_rouge1": 0, "avg_rouge2": 0, "avg_rougeL": 0, "num_reports": 0}

    avg_rouge1 = sum(r["rouge1"] for r in results) / n
    avg_rouge2 = sum(r["rouge2"] for r in results) / n
    avg_rougeL = sum(r["rougeL"] for r in results) / n

    print(f"\n{'=' * 50}")
    print("Information Fidelity Evaluation Results (description only)")
    print(f"{'=' * 50}")
    print(f"Reports evaluated: {n}")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")

    # Build output
    output = {
        "avg_rouge1": avg_rouge1,
        "avg_rouge2": avg_rouge2,
        "avg_rougeL": avg_rougeL,
        "num_reports": n,
        "details": results,
    }

    # Save results
    if output_path is None:
        output_path = Path(reconstructed_path).parent / "eval_info_fidelity.json"
    save_json(output, Path(output_path))

    return output


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate information fidelity (ROUGE)",
    )
    parser.add_argument(
        "--reconstructed",
        required=True,
        help="Path to reconstructed_output.json",
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
        "--config",
        default=None,
        help="Path to config file",
    )
    args = parser.parse_args()

    evaluate_info_fidelity(args.reconstructed, args.gt, args.output, args.config)


if __name__ == "__main__":
    main()
