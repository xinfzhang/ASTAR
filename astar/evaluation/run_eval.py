"""
Evaluation Suite Entry Point

Complete evaluation pipeline:
1. Template Quality (Cov_case, Cov_key)
2. Reconstruct Reports (from structured data)
3. Information Fidelity (ROUGE-1/2/L, description only)
4. Text Similarity (chrF, chrF++, BERTScore)
5. Diagnostic Fidelity (PDA, KFP, CA)

Input:
- template_final.json: Report template
- structured_output.json: Structuring module output
- gt.csv: Original reports (Ground Truth)
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

from astar.core.config import get_config, load_config
from astar.core.utils import save_json

from astar.evaluation.template_quality import evaluate_template_quality
from astar.evaluation.reconstruct import reconstruct_reports
from astar.evaluation.info_fidelity import evaluate_info_fidelity
from astar.evaluation.text_similarity import evaluate_text_similarity
from astar.evaluation.diag_fidelity import evaluate_diag_fidelity


def run_all_evaluations(
    template_path: str,
    structured_path: str,
    gt_path: str,
    output_dir: Optional[str] = None,
    num_reports: Optional[int] = None,
    max_workers: Optional[int] = None,
    config_path: Optional[str] = None,
    skip_reconstruct: bool = False,
    skip_bert: bool = False,
) -> Dict:
    """
    Run all evaluation metrics.

    Args:
        template_path: Path to template_final.json.
        structured_path: Path to structured_output.json.
        gt_path: Path to GT CSV file.
        output_dir: Output directory for results.
        num_reports: Number of reports to evaluate.
        max_workers: Number of parallel workers.
        config_path: Path to config file.
        skip_reconstruct: Skip reconstruction step (use existing results).
        skip_bert: Skip BERTScore evaluation (slow).

    Returns:
        Dictionary with all evaluation results.
    """
    # Load configuration
    if config_path:
        load_config(config_path)
    cfg = get_config()

    if max_workers is None:
        max_workers = cfg.evaluation.max_workers
    if num_reports is None:
        num_reports = cfg.evaluation.num_reports

    print("=" * 60)
    print("ASTAR Evaluation Suite")
    print("=" * 60)
    print(f"Template: {template_path}")
    print(f"Structured: {structured_path}")
    print(f"GT: {gt_path}")
    print()

    if output_dir is None:
        output_dir = Path(structured_path).parent
    else:
        output_dir = Path(output_dir)

    results = {}

    # 1. Template Quality
    print("\n" + "=" * 60)
    print("[1/5] Evaluating Template Quality")
    print("=" * 60)
    try:
        tq_result = evaluate_template_quality(
            template_path,
            gt_path,
            output_path=str(output_dir / "eval_template_quality.json"),
            num_reports=num_reports,
            max_workers=max_workers,
        )
        if tq_result and tq_result.get("num_reports", 0) > 0:
            results["template_quality"] = {
                "cov_case": tq_result["cov_case"],
                "cov_key": tq_result["cov_key"],
            }
        else:
            results["template_quality"] = None
    except Exception as e:
        print(f"Template Quality evaluation failed: {e}")
        results["template_quality"] = None

    # 2. Reconstruct Reports
    reconstructed_path = str(output_dir / "reconstructed_output.json")

    if skip_reconstruct and Path(reconstructed_path).exists():
        print("\n" + "=" * 60)
        print("[2/5] Skipping Report Reconstruction (using existing results)")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("[2/5] Reconstructing Reports")
        print("=" * 60)
        try:
            reconstruct_reports(
                structured_path,
                output_path=reconstructed_path,
                max_workers=max_workers,
            )
        except Exception as e:
            print(f"Report reconstruction failed: {e}")
            reconstructed_path = None

    # 3. Information Fidelity (ROUGE)
    print("\n" + "=" * 60)
    print("[3/5] Evaluating Information Fidelity (ROUGE)")
    print("=" * 60)
    try:
        if reconstructed_path and Path(reconstructed_path).exists():
            if_result = evaluate_info_fidelity(
                reconstructed_path,
                gt_path,
                output_path=str(output_dir / "eval_info_fidelity.json"),
            )
            if if_result and if_result.get("num_reports", 0) > 0:
                results["info_fidelity"] = {
                    "rouge1": if_result["avg_rouge1"],
                    "rouge2": if_result["avg_rouge2"],
                    "rougeL": if_result["avg_rougeL"],
                }
            else:
                results["info_fidelity"] = None
        else:
            print("Skipping: No reconstructed reports")
            results["info_fidelity"] = None
    except Exception as e:
        print(f"Information Fidelity evaluation failed: {e}")
        results["info_fidelity"] = None

    # 4. Text Similarity (chrF, BERTScore)
    print("\n" + "=" * 60)
    print("[4/5] Evaluating Text Similarity (chrF, BERTScore)")
    print("=" * 60)
    try:
        if reconstructed_path and Path(reconstructed_path).exists():
            ts_result = evaluate_text_similarity(
                reconstructed_path,
                gt_path,
                output_path=str(output_dir / "eval_text_similarity.json"),
                skip_bert=skip_bert,
            )
            if ts_result and ts_result.get("num_reports", 0) > 0:
                results["text_similarity"] = {}
                if "avg_chrf" in ts_result:
                    results["text_similarity"]["chrf"] = ts_result["avg_chrf"]
                    results["text_similarity"]["chrfpp"] = ts_result["avg_chrfpp"]
                if "avg_bertscore_macbert" in ts_result:
                    results["text_similarity"]["bertscore_macbert"] = ts_result[
                        "avg_bertscore_macbert"
                    ]
                if "avg_bertscore_roberta" in ts_result:
                    results["text_similarity"]["bertscore_roberta"] = ts_result[
                        "avg_bertscore_roberta"
                    ]
            else:
                results["text_similarity"] = None
        else:
            print("Skipping: No reconstructed reports")
            results["text_similarity"] = None
    except Exception as e:
        print(f"Text Similarity evaluation failed: {e}")
        results["text_similarity"] = None

    # 5. Diagnostic Fidelity
    print("\n" + "=" * 60)
    print("[5/5] Evaluating Diagnostic Fidelity")
    print("=" * 60)
    try:
        df_result = evaluate_diag_fidelity(
            structured_path,
            gt_path,
            output_path=str(output_dir / "eval_diag_fidelity.json"),
            max_workers=max_workers,
        )
        if df_result and df_result.get("num_reports", 0) > 0:
            results["diag_fidelity"] = {
                "pda": df_result["avg_pda"],
                "kfp": df_result["avg_kfp"],
                "ca": df_result["avg_ca"],
            }
        else:
            results["diag_fidelity"] = None
    except Exception as e:
        print(f"Diagnostic Fidelity evaluation failed: {e}")
        results["diag_fidelity"] = None

    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    if results["template_quality"]:
        print(f"\n[Template Quality]")
        print(f"  Cov_case: {results['template_quality']['cov_case']:.4f}")
        print(f"  Cov_key:  {results['template_quality']['cov_key']:.4f}")

    if results["info_fidelity"]:
        print(f"\n[Information Fidelity]")
        print(f"  ROUGE-1: {results['info_fidelity']['rouge1']:.4f}")
        print(f"  ROUGE-2: {results['info_fidelity']['rouge2']:.4f}")
        print(f"  ROUGE-L: {results['info_fidelity']['rougeL']:.4f}")

    if results.get("text_similarity"):
        print(f"\n[Text Similarity]")
        if "chrf" in results["text_similarity"]:
            print(f"  chrF:   {results['text_similarity']['chrf']:.4f}")
            print(f"  chrF++: {results['text_similarity']['chrfpp']:.4f}")
        if "bertscore_macbert" in results["text_similarity"]:
            print(
                f"  BERTScore-MacBERT:  {results['text_similarity']['bertscore_macbert']:.4f}"
            )
        if "bertscore_roberta" in results["text_similarity"]:
            print(
                f"  BERTScore-RoBERTa:  {results['text_similarity']['bertscore_roberta']:.4f}"
            )

    if results["diag_fidelity"]:
        print(f"\n[Diagnostic Fidelity]")
        print(f"  PDA: {results['diag_fidelity']['pda']:.4f}")
        print(f"  KFP: {results['diag_fidelity']['kfp']:.4f}")
        print(f"  CA:  {results['diag_fidelity']['ca']:.4f}")

    # Save summary
    save_json(results, output_dir / "eval_summary.json")

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="ASTAR Evaluation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all evaluations
  astar-eval --template output/template_final.json --structured output/structured_output.json --gt data/test.csv

  # Skip slow BERTScore
  astar-eval --template output/template_final.json --structured output/structured_output.json --gt data/test.csv --skip_bert

  # Use existing reconstructed reports
  astar-eval --template output/template_final.json --structured output/structured_output.json --gt data/test.csv --skip_reconstruct
        """,
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Path to template_final.json",
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
        "--output_dir",
        default=None,
        help="Output directory for results",
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
    parser.add_argument(
        "--skip_reconstruct",
        action="store_true",
        help="Skip reconstruction step (use existing results)",
    )
    parser.add_argument(
        "--skip_bert",
        action="store_true",
        help="Skip BERTScore evaluation (slow)",
    )
    args = parser.parse_args()

    run_all_evaluations(
        args.template,
        args.structured,
        args.gt,
        args.output_dir,
        args.num_reports,
        args.workers,
        args.config,
        args.skip_reconstruct,
        args.skip_bert,
    )


if __name__ == "__main__":
    main()
