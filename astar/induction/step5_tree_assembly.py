"""
Step 5: Template Tree Assembly

Merge skeleton and slot assignments to generate the complete template with slots.
"""

import copy
from typing import Dict, List

from astar.core.config import get_config
from astar.core.utils import load_json, load_jsonl, save_json


def run_step5(output_dir=None):
    """
    Execute Step 5: Build Template Tree.

    Reads template_skeleton_updated.json, slot_catalog.jsonl, and slot_assignments.jsonl.
    Produces template_with_slots.json.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    skeleton = load_json(output_dir / "template_skeleton_updated.json")
    slot_catalog = {
        s["slot_id"]: s
        for s in load_jsonl(output_dir / "slot_catalog.jsonl")
    }
    assignments = {
        a["slot_id"]: a
        for a in load_jsonl(output_dir / "slot_assignments.jsonl")
    }

    template = copy.deepcopy(skeleton)

    def get_or_create(tree: Dict, path: str) -> Dict:
        """Navigate to or create a path in the tree."""
        if not path:
            return tree
        node = tree
        for p in path.split("."):
            if p not in node:
                node[p] = {}
            node = node[p]
        return node

    # Mount slots
    for sid, assign in assignments.items():
        tier = assign.get("tier", "drop")
        path = assign.get("path", "")
        if tier == "drop" or not path:
            continue

        slot_info = slot_catalog.get(sid, {})
        slot_entry = {
            "slot_id": sid,
            "k": slot_info.get("k", ""),
            "k_zh": slot_info.get("k_zh", ""),
            "tau": slot_info.get("tau", "freetext"),
            "omega": slot_info.get("omega", []),
        }

        node = get_or_create(template, path)
        tier_key = f"_slots_{tier}"
        if tier_key not in node:
            node[tier_key] = []
        node[tier_key].append(slot_entry)

    # Save result (renamed from schema_with_slots to template_with_slots)
    save_json(template, output_dir / "template_with_slots.json")

    # Statistics
    def count_slots(n):
        c = sum(len(n.get(f"_slots_{t}", [])) for t in ["core", "extended"])
        for k, v in n.items():
            if not k.startswith("_") and isinstance(v, dict):
                c += count_slots(v)
        return c

    print(f"[Step5] Done. Template with {count_slots(template)} slots")

    return template


if __name__ == "__main__":
    run_step5()
