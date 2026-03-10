"""
Template Induction Pipeline (Stages I-III)

This package implements the ASTAR template induction pipeline:
- Stage I: Slot Induction (steps 0-2)
- Stage II: Template Assembly (steps 3-5)
- Stage III: Template Refinement (step 6)
"""

from astar.induction.pipeline import run_pipeline

__all__ = ["run_pipeline"]
