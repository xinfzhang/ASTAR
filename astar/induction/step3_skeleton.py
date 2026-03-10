"""
Step 3: Template Skeleton Building

Stage II Preparation: Induce high-level structure skeleton from report samples.
"""

import random
from pathlib import Path

import pandas as pd

from astar.core.config import get_config
from astar.core.llm import call_llm_json
from astar.core.utils import save_json

# System prompt for skeleton building
BUILD_SKELETON_SYSTEM = """You are a medical report structuring expert. Based on the given medical report samples, induce a high-level report structure skeleton.

Output JSON format (nested tree):
{
  "desc": {
    "fetus": {"head": {}, "face": {}, "spine": {}, ...},
    "maternal": {"uterus": {}, "adnexa": {}, ...}
  },
  "diag": {"primary_diagnosis": {}, "secondary_findings": {}}
}

Rules:
1. Top level divides into desc (description) and diag (diagnosis)
2. Organize hierarchy by anatomical structure or logical categories
3. Leaf nodes are empty objects {}"""


def run_step3(output_dir=None):
    """
    Execute Step 3: Build Template Skeleton.

    Reads sample reports and produces template_skeleton.json.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    sk_cfg = cfg.induction.skeleton

    # Load report samples
    df = pd.read_csv(cfg.data.train_csv, encoding="utf-8")
    df.columns = df.columns.str.strip()
    num = min(sk_cfg.num_sample_reports, len(df))
    indices = random.sample(range(len(df)), num)
    reports = [df.iloc[i].to_dict() for i in indices]
    print(f"[Step3] Sampled {len(reports)} reports")

    # Build prompt
    report_texts = []
    for i, r in enumerate(reports):
        desc = r.get(cfg.data.description_column, "")
        diag = r.get(cfg.data.diagnosis_column, "")
        report_texts.append(
            f"=== Report {i + 1} ===\n"
            f"Description:\n{desc}\n"
            f"Diagnosis:\n{diag}"
        )

    user_prompt = (
        "Please induce a structure skeleton from the following report samples:\n\n"
        + "\n\n".join(report_texts)
    )
    result = call_llm_json(BUILD_SKELETON_SYSTEM, user_prompt)

    # Default skeleton if LLM fails
    template_skeleton = result if result else {
        "desc": {"fetus": {}, "maternal": {}},
        "diag": {"diagnosis": {}},
    }

    # Save result (renamed from schema_bone to template_skeleton)
    save_json(template_skeleton, output_dir / "template_skeleton.json")
    print(f"[Step3] Done. Template skeleton saved.")

    return template_skeleton


if __name__ == "__main__":
    run_step3()
