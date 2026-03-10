"""
Step 4: Slot Assignment (Round 1 + Round 2)

Stage II: Mount slots onto the template skeleton.
- Round 1: Initial assignment, may propose new branches
- Round 2: Optimize paths on the updated skeleton
"""

import copy
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from astar.core.config import get_config
from astar.core.llm import call_llm_json
from astar.core.utils import load_json, load_jsonl, save_json, save_jsonl

# System prompt for Round 1 assignment
ASSIGN_SYSTEM_R1 = """You are a medical report structuring expert. Given a slot and a skeleton, decide where to mount it.

Output JSON:
{
  "decision": "attach/propose_new/drop",
  "tier": "core/extended/drop",
  "path": "desc.fetus.head",
  "new_branch": {"parent": "desc.fetus", "name": "umbilical_cord"},
  "reason": "explanation"
}

Rules:
- attach: Mount to existing node
- propose_new: Needs a new branch
- drop: Discard
- tier: core (essential), extended (optional), drop (discard)"""

# System prompt for Round 2 assignment
ASSIGN_SYSTEM_R2 = """You are a medical report structuring expert. Given a slot and the updated skeleton, decide the final path.

Output JSON:
{
  "tier": "core/extended/drop",
  "path": "desc.fetus.head",
  "reason": "explanation"
}

Rules: Cannot propose new nodes, must use existing paths."""


def get_paths(tree: Dict, prefix: str = "") -> List[str]:
    """Get all paths in the skeleton tree."""
    paths = []
    for k, v in tree.items():
        if not k.startswith("_"):
            path = f"{prefix}.{k}" if prefix else k
            paths.append(path)
            if isinstance(v, dict):
                paths.extend(get_paths(v, path))
    return paths


def assign_slot_r1(slot: Dict, paths: List[str]) -> Dict:
    """Round 1: Initial slot assignment."""
    omega = slot.get("omega", [])[:5]
    user_prompt = (
        f"Slot: {slot.get('k')} ({slot.get('k_zh', '')})\n"
        f"tau: {slot.get('tau', '')}\n"
        f"omega: {omega}\n\n"
        f"Skeleton paths:\n" + "\n".join(paths)
    )

    result = call_llm_json(ASSIGN_SYSTEM_R1, user_prompt)
    return {
        "slot_id": slot.get("slot_id", ""),
        "decision": result.get("decision", "drop") if result else "drop",
        "tier": result.get("tier", "drop") if result else "drop",
        "path": result.get("path", "") if result else "",
        "new_branch": result.get("new_branch") if result else None,
    }


def assign_slot_r2(slot: Dict, r1: Dict, paths: List[str]) -> Dict:
    """Round 2: Optimize slot path."""
    omega = slot.get("omega", [])[:5]
    user_prompt = (
        f"Slot: {slot.get('k')} ({slot.get('k_zh', '')})\n"
        f"R1 path: {r1.get('path', '')}\n\n"
        f"Skeleton paths:\n" + "\n".join(paths)
    )

    result = call_llm_json(ASSIGN_SYSTEM_R2, user_prompt)
    return {
        "slot_id": slot.get("slot_id", ""),
        "tier": result.get("tier", r1.get("tier", "drop")) if result else r1.get("tier", "drop"),
        "path": result.get("path", r1.get("path", "")) if result else r1.get("path", ""),
    }


def integrate_branch(tree: Dict, branch: Dict) -> bool:
    """Integrate a new branch into the skeleton."""
    if not branch:
        return False
    parent_path = branch.get("parent", "")
    new_name = branch.get("name", "")
    if not parent_path or not new_name:
        return False

    parts = parent_path.split(".")
    node = tree
    for p in parts:
        if p not in node:
            return False
        node = node[p]

    if new_name not in node:
        node[new_name] = {}
        return True
    return False


def run_step4(output_dir=None):
    """
    Execute Step 4: Slot Assignment.

    Reads template_skeleton.json and slot_catalog.jsonl.
    Produces template_skeleton_updated.json and slot_assignments.jsonl.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    sa_cfg = cfg.induction.slot_assignment

    skeleton = load_json(output_dir / "template_skeleton.json")
    slots = load_jsonl(output_dir / "slot_catalog.jsonl")
    print(f"[Step4] Loaded {len(slots)} slots")

    # === Round 1 ===
    print("[Step4] Round 1: Initial assignment")
    paths = get_paths(skeleton)
    r1_results = {}

    with ThreadPoolExecutor(max_workers=sa_cfg.round1_max_workers) as executor:
        futures = {
            executor.submit(assign_slot_r1, s, paths): s["slot_id"]
            for s in slots
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="R1"):
            sid = futures[fut]
            try:
                r1_results[sid] = fut.result()
                # Incrementally integrate new branches
                if r1_results[sid].get("new_branch"):
                    integrate_branch(skeleton, r1_results[sid]["new_branch"])
            except Exception as e:
                r1_results[sid] = {"slot_id": sid, "tier": "drop", "path": ""}

    # Save updated skeleton
    save_json(skeleton, output_dir / "template_skeleton_updated.json")

    # === Round 2 ===
    print("[Step4] Round 2: Path optimization")
    paths = get_paths(skeleton)  # Updated paths
    final_assignments = []

    active_slots = [
        s for s in slots
        if r1_results.get(s["slot_id"], {}).get("tier") != "drop"
    ]

    with ThreadPoolExecutor(max_workers=sa_cfg.round2_max_workers) as executor:
        futures = {
            executor.submit(assign_slot_r2, s, r1_results[s["slot_id"]], paths): s["slot_id"]
            for s in active_slots
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="R2"):
            try:
                final_assignments.append(fut.result())
            except Exception as e:
                pass

    # Add dropped slots
    for s in slots:
        if r1_results.get(s["slot_id"], {}).get("tier") == "drop":
            final_assignments.append({
                "slot_id": s["slot_id"],
                "tier": "drop",
                "path": "",
            })

    save_jsonl(final_assignments, output_dir / "slot_assignments.jsonl")

    stats = {"core": 0, "extended": 0, "drop": 0}
    for a in final_assignments:
        stats[a.get("tier", "drop")] += 1
    print(f"[Step4] Done. core={stats['core']}, extended={stats['extended']}, drop={stats['drop']}")

    return final_assignments


if __name__ == "__main__":
    run_step4()
