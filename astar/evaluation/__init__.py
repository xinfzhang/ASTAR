"""
Evaluation Suite

This package provides comprehensive evaluation metrics:
- Template Quality: Cov_case, Cov_key
- Information Fidelity: ROUGE-1/2/L
- Text Similarity: chrF, chrF++, BERTScore
- Diagnostic Fidelity: PDA, KFP, CA
"""

from astar.evaluation.run_eval import run_all_evaluations

__all__ = ["run_all_evaluations"]
