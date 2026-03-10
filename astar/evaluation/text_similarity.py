"""
Text Similarity Evaluation

Evaluate text similarity between reconstructed and original reports:
- chrF / chrF++ (character-level n-gram)
- BERTScore (MacBERT, RoBERTa)

Input:
  - reconstructed_output.json (reconstructed descriptions)
  - gt.csv (original reports)
Output:
  - chrF, chrF++, BERTScore metrics
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# chrF
try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False
    print("Warning: sacrebleu not installed, run: pip install sacrebleu")

# BERTScore
try:
    import torch
    import bert_score.utils

    # Monkey patch to fix OverflowError with large model_max_length
    def patched_sent_encode(tokenizer, sent):
        sent = sent.strip()
        if sent == "":
            return tokenizer.build_inputs_with_special_tokens([])
        return tokenizer.encode(
            sent, add_special_tokens=True, max_length=512, truncation=True
        )
    bert_score.utils.sent_encode = patched_sent_encode

    from bert_score import score as bert_score_func
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False
    print("Warning: bert_score not installed, run: pip install bert-score torch")

from astar.core.config import get_config, load_config
from astar.core.utils import load_json, save_json


@dataclass
class BertModelConfig:
    """BERT model configuration."""
    name: str
    path: str
    num_layers: int = 12


DEFAULT_BERT_MODELS = {
    "MacBERT": BertModelConfig(
        name="MacBERT",
        path="hfl/chinese-macbert-base",
        num_layers=12,
    ),
    "RoBERTa": BertModelConfig(
        name="RoBERTa",
        path="hfl/chinese-roberta-wwm-ext",
        num_layers=12,
    ),
}


def compute_chrf(
    references: List[str],
    hypotheses: List[str],
) -> Dict[str, List[float]]:
    """
    Compute chrF and chrF++ scores.

    Args:
        references: Reference texts.
        hypotheses: Hypothesis texts.

    Returns:
        Dictionary with chrF and chrF++ score lists.
    """
    if not HAS_SACREBLEU:
        print("Warning: sacrebleu not installed, cannot compute chrF")
        return {"chrf": [], "chrfpp": []}

    chrf_scores = []
    chrfpp_scores = []

    for ref, hyp in zip(references, hypotheses):
        if not ref or not hyp:
            chrf_scores.append(0.0)
            chrfpp_scores.append(0.0)
            continue

        # chrF (word_order=0, character-level only)
        chrf_res = sacrebleu.sentence_chrf(hyp, [ref])
        chrf_scores.append(chrf_res.score / 100.0)

        # chrF++ (word_order=2, includes word-level)
        chrfpp_res = sacrebleu.sentence_chrf(hyp, [ref], word_order=2)
        chrfpp_scores.append(chrfpp_res.score / 100.0)

    return {"chrf": chrf_scores, "chrfpp": chrfpp_scores}


def compute_bertscore(
    references: List[str],
    hypotheses: List[str],
    model_config: BertModelConfig,
    batch_size: int = 16,
) -> Optional[List[float]]:
    """
    Compute BERTScore.

    Args:
        references: Reference texts.
        hypotheses: Hypothesis texts.
        model_config: BERT model configuration.
        batch_size: Batch size for processing.

    Returns:
        List of F1 scores, or None on failure.
    """
    if not HAS_BERTSCORE:
        print("Warning: bert_score not installed, cannot compute BERTScore")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[BERTScore] Using device: {device}")
    print(f"[BERTScore] Model: {model_config.name} ({model_config.path})")

    try:
        P, R, F1 = bert_score_func(
            hypotheses,
            references,
            model_type=model_config.path,
            num_layers=model_config.num_layers,
            verbose=True,
            device=device,
            batch_size=batch_size,
        )
        return F1.tolist()
    except Exception as e:
        print(f"[BERTScore] Error with {model_config.name}: {e}")
        return None


def evaluate_text_similarity(
    reconstructed_path: str,
    gt_path: str,
    output_path: str = None,
    config_path: str = None,
    bert_models: Dict[str, str] = None,
    skip_bert: bool = False,
) -> Dict:
    """
    Evaluate text similarity metrics.

    Args:
        reconstructed_path: Path to reconstructed_output.json.
        gt_path: Path to GT CSV file.
        output_path: Output path for results.
        config_path: Path to config file.
        bert_models: Custom BERT model paths.
        skip_bert: Whether to skip BERTScore evaluation.

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
    print(f"[TextSimilarity] Loaded {len(results_list)} reconstructed reports")

    success_count = sum(1 for r in results_list if r.get("success", False))
    print(f"[TextSimilarity] Successfully reconstructed: {success_count}")

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
        idx_id = str(idx)
        id_to_gt_desc[idx_id] = str(row.get(desc_column, "") or "")

    # Collect paired data
    references = []
    hypotheses = []
    ids = []

    for item in results_list:
        rec_id = str(item.get("id", ""))

        if not item.get("success", False):
            continue

        reconstructed_desc = item.get("reconstructed_desc", "")
        gt_desc = id_to_gt_desc.get(rec_id, "")

        if not gt_desc:
            print(f"Warning: ID {rec_id} not found in GT")
            continue

        references.append(gt_desc)
        hypotheses.append(reconstructed_desc)
        ids.append(rec_id)

    n = len(references)
    if n == 0:
        print("No successful results to evaluate")
        return {"num_reports": 0}

    print(f"[TextSimilarity] Evaluating {n} reports")

    results = {"num_reports": n, "details": []}

    # 1. Compute chrF
    print("\n[1/2] Computing chrF / chrF++...")
    chrf_results = compute_chrf(references, hypotheses)

    if chrf_results["chrf"]:
        avg_chrf = np.mean(chrf_results["chrf"])
        avg_chrfpp = np.mean(chrf_results["chrfpp"])
        std_chrf = np.std(chrf_results["chrf"])
        std_chrfpp = np.std(chrf_results["chrfpp"])

        results["avg_chrf"] = avg_chrf
        results["avg_chrfpp"] = avg_chrfpp
        results["std_chrf"] = std_chrf
        results["std_chrfpp"] = std_chrfpp

        print(f"  chrF:   {avg_chrf:.4f} +/- {std_chrf:.4f}")
        print(f"  chrF++: {avg_chrfpp:.4f} +/- {std_chrfpp:.4f}")

    # 2. Compute BERTScore
    if not skip_bert:
        print("\n[2/2] Computing BERTScore...")

        models_to_use = DEFAULT_BERT_MODELS.copy()
        if bert_models:
            for name, path in bert_models.items():
                models_to_use[name] = BertModelConfig(name=name, path=path)

        for model_name, model_cfg in models_to_use.items():
            print(f"\n  Computing {model_name}...")
            f1_scores = compute_bertscore(references, hypotheses, model_cfg)

            if f1_scores:
                avg_f1 = np.mean(f1_scores)
                std_f1 = np.std(f1_scores)

                results[f"avg_bertscore_{model_name.lower()}"] = avg_f1
                results[f"std_bertscore_{model_name.lower()}"] = std_f1

                print(f"  BERTScore-{model_name} F1: {avg_f1:.4f} +/- {std_f1:.4f}")

                # Save detailed scores
                for i, (rec_id, score) in enumerate(zip(ids, f1_scores)):
                    if i >= len(results["details"]):
                        results["details"].append({"id": rec_id})
                    results["details"][i][f"bertscore_{model_name.lower()}"] = score
    else:
        print("\n[2/2] Skipping BERTScore (--skip_bert)")

    # Add chrF detailed scores
    for i, (rec_id, chrf, chrfpp) in enumerate(
        zip(ids, chrf_results["chrf"], chrf_results["chrfpp"])
    ):
        if i >= len(results["details"]):
            results["details"].append({"id": rec_id})
        results["details"][i]["chrf"] = chrf
        results["details"][i]["chrfpp"] = chrfpp

    # Print summary
    print(f"\n{'=' * 50}")
    print("Text Similarity Evaluation Results")
    print(f"{'=' * 50}")
    print(f"Reports evaluated: {n}")
    if "avg_chrf" in results:
        print(f"chrF:   {results['avg_chrf']:.4f}")
        print(f"chrF++: {results['avg_chrfpp']:.4f}")
    for key in results:
        if key.startswith("avg_bertscore_"):
            model_name = key.replace("avg_bertscore_", "").upper()
            print(f"BERTScore-{model_name}: {results[key]:.4f}")

    # Save results
    if output_path is None:
        output_path = Path(reconstructed_path).parent / "eval_text_similarity.json"
    save_json(results, Path(output_path))

    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate text similarity (chrF, BERTScore)",
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
    parser.add_argument(
        "--skip_bert",
        action="store_true",
        help="Skip BERTScore evaluation",
    )
    parser.add_argument(
        "--macbert_path",
        default=None,
        help="MacBERT model path",
    )
    parser.add_argument(
        "--roberta_path",
        default=None,
        help="RoBERTa model path",
    )
    args = parser.parse_args()

    bert_models = {}
    if args.macbert_path:
        bert_models["MacBERT"] = args.macbert_path
    if args.roberta_path:
        bert_models["RoBERTa"] = args.roberta_path

    evaluate_text_similarity(
        args.reconstructed,
        args.gt,
        args.output,
        args.config,
        bert_models if bert_models else None,
        args.skip_bert,
    )


if __name__ == "__main__":
    main()
