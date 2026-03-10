"""
Template Induction Pipeline Orchestrator

This module provides the main entry point for running the complete
ASTAR template induction pipeline.

The pipeline consists of:
- Stage I: Slot Induction (Steps 0-2)
- Stage II: Template Assembly (Steps 3-5)
- Stage III: Template Refinement (Step 6)

Usage:
    from astar.induction import run_pipeline
    run_pipeline()

Or from command line:
    python -m astar.induction.pipeline
    astar-induction  # if installed as package
"""

import argparse
import time
from pathlib import Path
from typing import Optional, List

from astar.core.config import get_config, load_config

from astar.induction.step0_span_clustering import run_step0
from astar.induction.step1_slot_induction import run_step1
from astar.induction.step2_slot_merging import run_step2
from astar.induction.step3_skeleton import run_step3
from astar.induction.step4_slot_assignment import run_step4
from astar.induction.step5_tree_assembly import run_step5
from astar.induction.step6_refinement import run_step6


STEP_FUNCTIONS = {
    0: ("Span Clustering", run_step0),
    1: ("Slot Induction", run_step1),
    2: ("Slot Merging", run_step2),
    3: ("Skeleton Building", run_step3),
    4: ("Slot Assignment", run_step4),
    5: ("Tree Assembly", run_step5),
    6: ("Refinement", run_step6),
}


def run_pipeline(
    config_path: Optional[str] = None,
    steps: Optional[List[int]] = None,
    start_step: int = 0,
    end_step: int = 6,
) -> Path:
    """
    Run the complete template induction pipeline.

    Args:
        config_path: Path to config.yaml file.
        steps: Specific steps to run (e.g., [0, 1, 2]). If None, runs range.
        start_step: First step to run (default: 0).
        end_step: Last step to run (default: 6).

    Returns:
        Path to the output directory.
    """
    # Load configuration
    if config_path:
        load_config(config_path)
    cfg = get_config()

    output_dir = cfg.ensure_output_dir()
    print("=" * 60)
    print("ASTAR Template Induction Pipeline")
    print("=" * 60)
    print(f"Experiment: {cfg.experiment_id}")
    print(f"Output: {output_dir}")
    print(f"Model: {cfg.llm.model}")
    print()

    # Determine which steps to run
    if steps:
        steps_to_run = sorted(steps)
    else:
        steps_to_run = list(range(start_step, end_step + 1))

    print(f"Running steps: {steps_to_run}")
    print()

    total_start = time.time()

    # Run each step
    for step in steps_to_run:
        if step not in STEP_FUNCTIONS:
            print(f"[Warning] Unknown step {step}, skipping")
            continue

        step_name, step_func = STEP_FUNCTIONS[step]
        print("=" * 60)
        print(f"Step {step}: {step_name}")
        print("=" * 60)

        step_start = time.time()
        try:
            step_func(output_dir)
        except Exception as e:
            print(f"[Error] Step {step} failed: {e}")
            raise

        step_time = time.time() - step_start
        print(f"[Step {step}] Completed in {step_time:.1f}s")
        print()

    total_time = time.time() - total_start
    print("=" * 60)
    print(f"Pipeline completed in {total_time:.1f}s")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    return output_dir


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="ASTAR Template Induction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  astar-induction

  # Run specific steps
  astar-induction --steps 0 1 2

  # Run from step 3 onwards
  astar-induction --start 3

  # Use custom config
  astar-induction --config path/to/config.yaml
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Specific steps to run (e.g., --steps 0 1 2)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="First step to run (default: 0)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=6,
        help="Last step to run (default: 6)",
    )
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        steps=args.steps,
        start_step=args.start,
        end_step=args.end,
    )


if __name__ == "__main__":
    main()
