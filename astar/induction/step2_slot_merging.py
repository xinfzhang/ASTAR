"""
Step 2: Slot Merging

Stage I - Part 3: Globally merge slots by standardized key name k,
combine omega sets, and unify tau types.
"""

from typing import Dict, List, Any
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

# System prompt for slot merging
MERGE_SLOTS_SYSTEM = """You are a medical report structuring expert. Given a group of slots with the same standardized key name (k), please merge them into one canonical slot.

Tasks:
1. Merge allowed value sets (omega)
2. Unify value type (tau): use majority vote

Output JSON format:
{
  "slot_id": "unique identifier (based on k)",
  "k": "standardized key name",
  "k_zh": "Chinese key name",
  "tau": "categorical/numerical/freetext",
  "omega": ["merged allowed values"],
  "description": "semantic description"
}"""


def merge_slots_by_key(k: str, members: List[Dict]) -> Dict[str, Any]:
    """
    Merge slots with the same key k.

    Args:
        k: Standardized key name.
        members: List of slot dictionaries to merge.

    Returns:
        Merged canonical slot.
    """
    # Merge omega sets
    all_omega = list(set(v for m in members for v in m.get("omega", [])))

    # Majority vote for tau
    tau_counts = defaultdict(int)
    for m in members:
        tau_counts[m.get("tau", "freetext")] += 1
    majority_tau = max(tau_counts.keys(), key=lambda t: tau_counts[t])

    # Most common k_zh
    k_zh_counts = defaultdict(int)
    for m in members:
        if m.get("k_zh"):
            k_zh_counts[m["k_zh"]] += 1
    k_zh = max(k_zh_counts.keys(), key=lambda x: k_zh_counts[x]) if k_zh_counts else ""

    # Collect examples
    examples = []
    for m in members:
        for span in m.get("example_spans", [])[:2]:
            val = span.get("a_val", "") if isinstance(span, dict) else span
            if val and val not in examples:
                examples.append(val)

    # LLM refinement
    member_info = [
        f"cluster={m.get('cluster_id')} | k_zh={m.get('k_zh', '')} | "
        f"tau={m.get('tau', '')} | omega={m.get('omega', [])[:5]}"
        for m in members[:10]
    ]
    user_prompt = (
        f"Key name: {k}\n"
        f"Members ({len(members)}):\n"
        + "\n".join(member_info)
        + f"\n\nCandidate merged omega: {all_omega[:15]}\n"
        f"tau statistics: {dict(tau_counts)}"
    )

    result = call_llm_json(MERGE_SLOTS_SYSTEM, user_prompt)

    slot = {
        "slot_id": f"slot_{k}",
        "k": k,
        "k_zh": k_zh,
        "tau": majority_tau,
        "omega": all_omega[:20],
        "description": "",
        "canonical_examples": examples[:10],
        "source_clusters": list(set(m.get("cluster_id", 0) for m in members)),
    }

    # Override with LLM result if available
    if result:
        for key in ["slot_id", "k", "k_zh", "tau", "omega", "description"]:
            if result.get(key):
                slot[key] = result[key]

    return slot


def run_step2(output_dir=None):
    """
    Execute Step 2: Global Slot Merging.

    Reads induced_slots_step1.jsonl and produces slot_catalog.jsonl.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    sm_cfg = cfg.induction.slot_merging

    # Load induced slots from step 1
    slots = load_jsonl(output_dir / "induced_slots_step1.jsonl")
    print(f"[Step2] Loaded {len(slots)} induced slots")

    # Group by k
    groups = defaultdict(list)
    for s in slots:
        k = s.get("k", "").lower().strip()
        if k:
            groups[k].append(s)

    keys = list(groups.keys())
    if sm_cfg.max_families:
        keys = keys[:sm_cfg.max_families]
    print(f"[Step2] Found {len(keys)} unique keys")

    # Merge in parallel
    canonical_slots = []
    with ThreadPoolExecutor(max_workers=sm_cfg.max_workers) as executor:
        futures = {
            executor.submit(merge_slots_by_key, k, groups[k]): k
            for k in keys
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Merging slots"):
            try:
                canonical_slots.append(fut.result())
            except Exception as e:
                print(f"[Step2] Error: {e}")

    canonical_slots.sort(key=lambda s: s.get("slot_id", ""))

    # Save results
    save_jsonl(canonical_slots, output_dir / "slot_catalog.jsonl")
    print(f"[Step2] Done. {len(canonical_slots)} canonical slots")

    return canonical_slots


if __name__ == "__main__":
    run_step2()
