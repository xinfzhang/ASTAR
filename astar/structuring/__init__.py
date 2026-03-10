"""
Report Structuring Module

This package provides utilities for:
- Structuring reports using induced templates
- Template traversal and manipulation
"""

from astar.structuring.template_utils import (
    flatten_template,
    extract_template_slots,
    extract_template_keys,
    unflatten_result,
)

__all__ = [
    "flatten_template",
    "extract_template_slots",
    "extract_template_keys",
    "unflatten_result",
]
