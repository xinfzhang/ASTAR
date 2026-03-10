# ASTAR: Automatic Template Induction for Medical Report Structuring

ASTAR is a three-stage pipeline for automatic template induction from medical reports. Given a collection of free-text medical reports, ASTAR induces a structured template that captures the key clinical concepts, then uses this template to structure new reports.

## Features

- **Automatic Template Induction**: Learn report structure from data without manual template design
- **Three-Stage Pipeline**: Slot induction, template assembly, and refinement
- **Comprehensive Evaluation**: Template quality, information fidelity, and diagnostic fidelity metrics
- **Modular Design**: Easy to extend and customize for different medical domains

## Installation

```bash
# Clone the repository
git clone https://github.com/xinfzhang/ASTAR.git
cd ASTAR

# Install in development mode
pip install -e .

# Install with evaluation dependencies
pip install -e ".[eval]"

# Install all dependencies
pip install -e ".[all]"
```

## Quick Start

### 1. Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env
# Edit .env and set ASTAR_API_KEY=your_api_key_here
```

Edit `config.yaml` to configure:
- LLM endpoint and model
- Data paths
- Pipeline parameters

### 2. Template Induction

```bash
# Run the full template induction pipeline
astar-induction --config config.yaml

# Or run specific steps
astar-induction --steps 0 1 2  # Run only Stage I (slot induction)
astar-induction --start 3      # Start from Stage II
```

### 3. Report Structuring

```bash
# Structure reports using the induced template
python -m astar.structuring.structure \
    --data data/test.csv \
    --template output/exp_001/template_final.json \
    --output output/structured_output.json
```

### 4. Evaluation

```bash
# Run all evaluation metrics
astar-eval \
    --template output/exp_001/template_final.json \
    --structured output/structured_output.json \
    --gt data/test.csv

# Skip slow BERTScore evaluation
astar-eval ... --skip_bert
```

## Pipeline Overview

### Stage I: Slot Induction (Steps 0-2)

1. **Step 0 - Span Clustering**: Extract clinical concept spans from reports and cluster them
2. **Step 1 - Slot Induction**: Refine clusters to induce normalized slots (k, tau, omega)
3. **Step 2 - Slot Merging**: Globally merge slots by key name

### Stage II: Template Assembly (Steps 3-5)

4. **Step 3 - Skeleton Building**: Induce high-level structure from sample reports
5. **Step 4 - Slot Assignment**: Mount slots onto the skeleton (two-round process)
6. **Step 5 - Tree Assembly**: Merge skeleton and slots into complete template

### Stage III: Template Refinement (Step 6)

7. **Step 6 - Refinement**: Intra-node reorganization and global harmonization

## Output Files

After running the pipeline, the output directory contains:

```
output/exp_001/
├── spans_with_cluster.jsonl     # Step 0: Extracted spans
├── induced_slots_step1.jsonl    # Step 1: Induced slots
├── slot_catalog.jsonl           # Step 2: Merged slots
├── template_skeleton.json       # Step 3: Structure skeleton
├── template_skeleton_updated.json  # Step 4: Updated skeleton
├── slot_assignments.jsonl       # Step 4: Slot assignments
├── template_with_slots.json     # Step 5: Template with slots
└── template_final.json          # Step 6: Final refined template
```

## Evaluation Metrics

- **Template Quality**: Cov_case (case-level coverage), Cov_key (key-level coverage)
- **Information Fidelity**: ROUGE-1/2/L scores
- **Text Similarity**: chrF, chrF++, BERTScore
- **Diagnostic Fidelity**: PDA, KFP, CA

## Project Structure

```
astar/
├── astar/
│   ├── core/           # Shared infrastructure
│   │   ├── config.py   # Unified configuration
│   │   ├── llm.py      # LLM API client
│   │   └── utils.py    # File I/O utilities
│   ├── induction/      # Template induction pipeline
│   │   ├── step0_span_clustering.py
│   │   ├── step1_slot_induction.py
│   │   ├── step2_slot_merging.py
│   │   ├── step3_skeleton.py
│   │   ├── step4_slot_assignment.py
│   │   ├── step5_tree_assembly.py
│   │   ├── step6_refinement.py
│   │   └── pipeline.py
│   ├── structuring/    # Report structuring
│   │   ├── structure.py
│   │   └── template_utils.py
│   └── evaluation/     # Evaluation suite
│       ├── run_eval.py
│       ├── reconstruct.py
│       ├── template_quality.py
│       ├── info_fidelity.py
│       ├── text_similarity.py
│       └── diag_fidelity.py
├── examples/
│   ├── example_report.csv
│   └── example_template.json
├── config.yaml
├── pyproject.toml
└── README.md
```

## Configuration Reference

See `config.yaml` for all available options. Key settings:

```yaml
llm:
  base_url: "https://api.openai.com/v1"  # OpenAI-compatible endpoint
  model: "gpt-4o-mini"
  embedding_model: "text-embedding-3-small"

data:
  train_csv: "./data/train.csv"
  description_column: "description"
  diagnosis_column: "diagnosis"
```

## Python API

```python
from astar.core.config import get_config, load_config
from astar.induction import run_pipeline
from astar.structuring.structure import structure_reports
from astar.evaluation import run_all_evaluations

# Load custom config
load_config("path/to/config.yaml")

# Run template induction
output_dir = run_pipeline()

# Structure reports
results = structure_reports(
    data_path="data/test.csv",
    template_path=f"{output_dir}/template_final.json",
)

# Evaluate
metrics = run_all_evaluations(
    template_path=f"{output_dir}/template_final.json",
    structured_path="output/structured_output.json",
    gt_path="data/test.csv",
)
```

## Requirements

- Python >= 3.9
- See `pyproject.toml` for full dependency list

## License

MIT License

## Citation

If you use ASTAR in your research, please cite:

```bibtex
@article{astar2026,
  title={ASTAR: Automatic Template Induction for Medical Report Structuring},
  author={...},
  journal={...},
  year={2026}
}
```
