"""
Step 6: Template Refinement (Stage III)

Stage III includes two operations:
1. Intra-node Reorganization: Merge overlapping slots, harmonize value types
2. Global Harmonization: Unify terminology across nodes
"""

import copy
from typing import Dict, List
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from astar.core.config import get_config
from astar.core.llm import call_llm_json
from astar.core.utils import load_json, load_jsonl, save_json

# System prompt for intra-node reorganization
INTRA_NODE_SYSTEM = """You are a medical report structuring expert. Given a list of slots under one node, please:
1. Merge slots with highly overlapping allowed values
2. Harmonize value types (majority vote principle)

Output JSON:
{
  "slots": [
    {
      "slot_id": "id",
      "k": "key name",
      "k_zh": "Chinese name",
      "tau": "type",
      "omega": ["values"],
      "merged_from": ["original ids"]
    }
  ]
}"""


def reorganize_node(path: str, slots: List[Dict]) -> List[Dict]:
    """
    Reorganize slots within a single node.

    Args:
        path: Node path in the template.
        slots: List of slots at this node.

    Returns:
        Reorganized list of slots.
    """
    if len(slots) <= 1:
        return slots

    slot_info = "\n".join([
        f"- {s.get('slot_id')}: {s.get('k_zh', '')} | "
        f"tau={s.get('tau', '')} | omega={s.get('omega', [])[:5]}"
        for s in slots[:15]
    ])
    user_prompt = f"Node: {path}\n\nSlots:\n{slot_info}"

    result = call_llm_json(INTRA_NODE_SYSTEM, user_prompt)
    return result.get("slots", slots) if result else slots


def harmonize_global(template: Dict) -> Dict:
    """
    Global harmonization: Unify the same concept across nodes.

    Args:
        template: Template tree.

    Returns:
        Harmonized template.
    """
    # Collect all slots grouped by k
    groups = defaultdict(list)

    def collect(node, path):
        for tier in ["core", "extended"]:
            for s in node.get(f"_slots_{tier}", []):
                k = s.get("k", "").lower()
                if k:
                    groups[k].append((path, tier, s))
        for key, val in node.items():
            if not key.startswith("_") and isinstance(val, dict):
                collect(val, f"{path}.{key}" if path else key)

    collect(template, "")

    # Unify multi-instance concepts
    for k, instances in groups.items():
        if len(instances) <= 1:
            continue
        canonical = instances[0][2]
        for _, _, slot in instances[1:]:
            if canonical.get("k_zh"):
                slot["k_zh"] = canonical["k_zh"]
            if canonical.get("omega"):
                slot["omega"] = canonical["omega"]
            if canonical.get("tau"):
                slot["tau"] = canonical["tau"]

    return template


def run_step6(output_dir=None):
    """
    Execute Step 6: Template Refinement.

    Reads template_with_slots.json and produces template_final.json.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    ref_cfg = cfg.induction.refinement

    template = load_json(output_dir / "template_with_slots.json")

    # Collect nodes that need processing
    nodes = []

    def collect_nodes(node, path):
        for tier in ["core", "extended"]:
            key = f"_slots_{tier}"
            if key in node and len(node[key]) > 1:
                nodes.append((path, tier, node, key))
        for k, v in node.items():
            if not k.startswith("_") and isinstance(v, dict):
                collect_nodes(v, f"{path}.{k}" if path else k)

    collect_nodes(template, "")

    # 1. Intra-node reorganization
    print("[Step6] Intra-node reorganization")
    with ThreadPoolExecutor(max_workers=ref_cfg.max_workers) as executor:
        futures = {
            executor.submit(reorganize_node, path, node[key]): (path, tier, node, key)
            for path, tier, node, key in nodes
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reorganizing"):
            path, tier, node, key = futures[fut]
            try:
                node[key] = fut.result()
            except Exception:
                pass

    # 2. Global harmonization
    print("[Step6] Global harmonization")
    template = harmonize_global(template)

    # Save final template (renamed from schema_final to template_final)
    save_json(template, output_dir / "template_final.json")
    print("[Step6] Done. Final template saved.")

    return template


if __name__ == "__main__":
    run_step6()
