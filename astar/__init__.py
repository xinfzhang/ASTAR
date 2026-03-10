"""
ASTAR: Automatic Template Induction for Medical Report Structuring

This package implements the ASTAR method for automatic template induction
from medical reports, as described in the accompanying paper.

Main components:
- core: Unified LLM client, configuration, and utilities
- induction: Template induction pipeline (Stages I-III)
- structuring: Report structuring using induced templates
- evaluation: Comprehensive evaluation suite

Usage:
    from astar.core.config import get_config
    from astar.core.llm import call_llm, call_llm_json
    from astar.induction import pipeline
"""

__version__ = "1.0.0"
__author__ = "ASTAR Team"

from astar.core.config import get_config, load_config
from astar.core.llm import call_llm, call_llm_json, get_embeddings

__all__ = [
    "get_config",
    "load_config",
    "call_llm",
    "call_llm_json",
    "get_embeddings",
]
