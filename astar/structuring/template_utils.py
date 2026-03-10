"""
Template Utilities

This module provides utilities for working with hierarchical templates:
- Flatten templates for LLM processing
- Extract slot information
- Unflatten results back to hierarchical structure

Key terminology (aligned with paper):
- "template" (not "schema"): The induced report structure
- "slot": A field in the template with (k, tau, omega) attributes
"""

import copy
from typing import Dict, List, Any, Tuple, Set


def flatten_template(template: Dict) -> Tuple[List[Dict], Dict[str, str]]:
    """
    Flatten hierarchical template to a flat slot list.

    Args:
        template: Hierarchical template dictionary.

    Returns:
        Tuple of (slots, key_to_path):
        - slots: Flat list of slots with {k, k_zh, tau, omega, path, tier}
        - key_to_path: Mapping from k to original path for reconstruction
    """
    slots = []
    key_to_path = {}

    def traverse(node: Dict, path: str = ""):
        for tier in ["_slots_core", "_slots_extended"]:
            if tier in node:
                for slot in node[tier]:
                    k = slot.get("k", "")
                    if not k:
                        continue

                    slot_info = {
                        "k": k,
                        "k_zh": slot.get("k_zh", k),
                        "tau": slot.get("tau", "str"),
                        "omega": slot.get("omega", []),
                        "path": path,
                        "tier": tier.replace("_slots_", ""),
                    }
                    slots.append(slot_info)
                    key_to_path[k] = path

        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                new_path = f"{path}.{key}" if path else key
                traverse(val, new_path)

    traverse(template)
    return slots, key_to_path


def extract_template_slots(template: Dict) -> List[Dict]:
    """
    Extract all slots from template with path information.

    Args:
        template: Hierarchical template dictionary.

    Returns:
        List of slot dictionaries with path and tier info.
    """
    slots = []

    def traverse(node: Dict, path: str = ""):
        for tier in ["_slots_core", "_slots_extended"]:
            if tier in node:
                for slot in node[tier]:
                    slot_copy = slot.copy()
                    slot_copy["path"] = path
                    slot_copy["tier"] = tier.replace("_slots_", "")
                    slots.append(slot_copy)
        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                new_path = f"{path}.{key}" if path else key
                traverse(val, new_path)

    traverse(template)
    return slots


def extract_template_keys(template: Dict) -> Set[str]:
    """
    Extract all slot keys (k and k_zh) from template.

    Args:
        template: Hierarchical template dictionary.

    Returns:
        Set of all slot keys.
    """
    keys = set()

    def traverse(node: Dict):
        for tier in ["_slots_core", "_slots_extended"]:
            if tier in node:
                for slot in node[tier]:
                    k = slot.get("k", "")
                    k_zh = slot.get("k_zh", "")
                    if k:
                        keys.add(k.lower())
                    if k_zh:
                        keys.add(k_zh)
        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                traverse(val)

    traverse(template)
    return keys


def slots_to_flat_template(
    slots: List[Dict],
    include_omega: bool = True,
) -> Dict[str, Any]:
    """
    Convert slots to flat template format for LLM filling.

    Format:
    {
        "field_name": {
            "description": "Chinese description",
            "type": "value type",
            "allowed_values": ["value1", ...],  # only if omega exists
            "value": null
        }
    }

    Args:
        slots: List of slot dictionaries.
        include_omega: Whether to include allowed values.

    Returns:
        Flat template dictionary.
    """
    template = {}
    for slot in slots:
        k = slot["k"]
        field_info = {
            "description": slot.get("k_zh", k),
            "type": slot.get("tau", "str"),
            "value": None,
        }
        omega = slot.get("omega", [])
        if include_omega and omega:
            # Limit omega length to avoid overly long prompts
            field_info["allowed_values"] = omega[:10]

        template[k] = field_info

    return template


def slots_to_simple_template(slots: List[Dict]) -> Dict[str, str]:
    """
    Convert slots to simple template format (key: description only).

    More concise, suitable for templates with many fields.

    Args:
        slots: List of slot dictionaries.

    Returns:
        Simple template dictionary {key: description}.
    """
    template = {}
    for slot in slots:
        k = slot["k"]
        k_zh = slot.get("k_zh", k)
        tau = slot.get("tau", "str")
        omega = slot.get("omega", [])

        desc = k_zh
        if tau and tau != "str":
            desc += f" [{tau}]"
        if omega:
            desc += f" (e.g., {', '.join(str(v) for v in omega[:3])})"

        template[k] = desc

    return template


def unflatten_result(
    flat_result: Dict,
    template: Dict,
    key_to_path: Dict[str, str],
) -> Dict:
    """
    Unflatten filled results back to hierarchical template structure.

    Args:
        flat_result: LLM-filled flat result {k: value}.
        template: Original hierarchical template.
        key_to_path: Mapping from key to path.

    Returns:
        Filled hierarchical template.
    """
    filled = copy.deepcopy(template)

    def set_slot_value(node: Dict, target_k: str, value: Any) -> bool:
        """Find and set slot value in node."""
        for tier in ["_slots_core", "_slots_extended"]:
            if tier in node:
                for slot in node[tier]:
                    if slot.get("k") == target_k:
                        slot["_value"] = value
                        return True
        return False

    def get_node_by_path(root: Dict, path: str) -> Dict:
        """Navigate to node by path."""
        if not path:
            return root
        parts = path.split(".")
        node = root
        for p in parts:
            if p in node:
                node = node[p]
            else:
                return root
        return node

    # Fill each value
    for k, value in flat_result.items():
        if value is None or value == "":
            continue

        path = key_to_path.get(k, "")
        node = get_node_by_path(filled, path)
        set_slot_value(node, k, value)

    return filled


def extract_filled_values(filled_template: Dict) -> Dict[str, Any]:
    """
    Extract all filled values from a template.

    Args:
        filled_template: Template with _value fields filled.

    Returns:
        Dictionary {k: value} of all filled values.
    """
    result = {}

    def traverse(node: Dict):
        for tier in ["_slots_core", "_slots_extended"]:
            if tier in node:
                for slot in node[tier]:
                    k = slot.get("k", "")
                    value = slot.get("_value")
                    if k and value is not None and value != "":
                        result[k] = value

        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                traverse(val)

    traverse(filled_template)
    return result


def format_slots_for_prompt(slots: List[Dict], max_slots: int = 50) -> str:
    """
    Format slots for inclusion in LLM prompts.

    Args:
        slots: List of slot dictionaries.
        max_slots: Maximum slots to include.

    Returns:
        Formatted string for prompt.
    """
    lines = []
    for slot in slots[:max_slots]:
        k = slot.get("k", "")
        k_zh = slot.get("k_zh", "")
        tau = slot.get("tau", "")
        omega = slot.get("omega", [])[:5]
        lines.append(f"- {k} ({k_zh}): tau={tau}, omega={omega}")
    return "\n".join(lines)


def get_template_structure_summary(template: Dict) -> Dict[str, Any]:
    """
    Generate a summary of template structure.

    Args:
        template: Hierarchical template dictionary.

    Returns:
        Summary with node counts, slot counts, etc.
    """
    summary = {
        "total_slots": 0,
        "core_slots": 0,
        "extended_slots": 0,
        "nodes": 0,
        "max_depth": 0,
    }

    def traverse(node: Dict, depth: int):
        summary["nodes"] += 1
        summary["max_depth"] = max(summary["max_depth"], depth)

        for tier, key in [("core", "_slots_core"), ("extended", "_slots_extended")]:
            count = len(node.get(key, []))
            summary["total_slots"] += count
            summary[f"{tier}_slots"] += count

        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                traverse(val, depth + 1)

    traverse(template, 0)
    return summary
