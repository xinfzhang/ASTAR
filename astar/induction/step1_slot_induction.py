"""
Step 1: Slot Induction

Stage I - Part 2: Refine each cluster to induce normalized slots phi = (k, tau, omega).

For each cluster, use LLM to subdivide spans into semantic subgroups
and induce a standardized slot for each subgroup.
"""

import random
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
from astar.core.utils import load_jsonl, save_jsonl

# System prompt for cluster refinement and slot induction
CLUSTER_REFINE_SYSTEM = """You are a medical report structuring expert. Given a list of spans from one cluster, please:

1. Subdivide spans by semantic consistency into subgroups
2. Induce a normalized slot (k, tau, omega) for each subgroup

Output JSON format:
{
  "subgroups": [
    {
      "group_id": "g1",
      "span_indices": [0, 2, 5],
      "slot": {
        "k": "standardized key name (English snake_case, e.g., fetal_bpd)",
        "k_zh": "Chinese key name (e.g., fetal biparietal diameter)",
        "tau": "categorical/numerical/freetext",
        "omega": ["allowed_value1", "allowed_value2"]
      },
      "description": "semantic description"
    }
  ]
}"""


def build_span_card(span: Dict) -> str:
    """Build a readable span description."""
    parts = []
    for key, label in [
        ("a_struct", "Structure"),
        ("a_sub", "Substructure"),
        ("a_attr", "Attribute"),
        ("a_val", "Value"),
        ("a_role", "Role"),
    ]:
        if span.get(key):
            parts.append(f"{label}:{span[key]}")
    return " | ".join(parts)


def refine_cluster(cluster_id: int, spans: List[Dict], max_spans: int) -> List[Dict]:
    """
    Refine a single cluster and induce slots.

    Args:
        cluster_id: Cluster identifier.
        spans: List of spans in this cluster.
        max_spans: Maximum spans to sample for LLM call.

    Returns:
        List of induced slots.
    """
    sampled = random.sample(spans, min(len(spans), max_spans))
    span_cards = [f"[{i}] {build_span_card(s)}" for i, s in enumerate(sampled)]
    user_prompt = (
        f"Cluster {cluster_id} ({len(spans)} spans, sampled {len(sampled)}):\n\n"
        + "\n".join(span_cards)
    )

    result = call_llm_json(CLUSTER_REFINE_SYSTEM, user_prompt)

    slots = []
    if result and "subgroups" in result:
        for sg in result["subgroups"]:
            slot = sg.get("slot", {})
            if slot.get("k"):
                indices = sg.get("span_indices", [])
                example_spans = [sampled[i] for i in indices if i < len(sampled)][:3]
                slots.append({
                    "cluster_id": cluster_id,
                    "group_id": sg.get("group_id", ""),
                    "k": slot.get("k", ""),
                    "k_zh": slot.get("k_zh", ""),
                    "tau": slot.get("tau", "freetext"),
                    "omega": slot.get("omega", []),
                    "description": sg.get("description", ""),
                    "example_spans": example_spans,
                })
    return slots


def run_step1(output_dir=None):
    """
    Execute Step 1: Cluster Refinement and Slot Induction.

    Reads spans_with_cluster.jsonl and produces induced_slots_step1.jsonl.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    si_cfg = cfg.induction.slot_induction

    # Load spans
    spans = load_jsonl(output_dir / "spans_with_cluster.jsonl")
    print(f"[Step1] Loaded {len(spans)} spans")

    # Group by cluster
    clusters = defaultdict(list)
    for s in spans:
        clusters[s.get("cluster_id", 0)].append(s)

    cluster_ids = sorted(clusters.keys())
    if si_cfg.max_clusters:
        cluster_ids = cluster_ids[:si_cfg.max_clusters]
    print(f"[Step1] Processing {len(cluster_ids)} clusters")

    # Process clusters in parallel
    all_slots = []
    with ThreadPoolExecutor(max_workers=si_cfg.max_workers) as executor:
        futures = {
            executor.submit(
                refine_cluster, cid, clusters[cid], si_cfg.max_spans_per_cluster
            ): cid
            for cid in cluster_ids
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Refining clusters"):
            try:
                all_slots.extend(fut.result())
            except Exception as e:
                print(f"[Step1] Error: {e}")

    # Save results
    save_jsonl(all_slots, output_dir / "induced_slots_step1.jsonl")
    print(f"[Step1] Done. Induced {len(all_slots)} slots")

    return all_slots


if __name__ == "__main__":
    run_step1()
