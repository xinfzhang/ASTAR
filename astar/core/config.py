"""
Unified Configuration Management

This module provides a single configuration system for all ASTAR components.
Configuration is loaded from config.yaml with environment variable overrides.

Usage:
    from astar.core.config import get_config
    cfg = get_config()
    print(cfg.llm.model)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import yaml


@dataclass
class LLMConfig:
    """LLM API configuration."""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.0
    max_retries: int = 3
    backoff_base: float = 10.0


@dataclass
class DataConfig:
    """Data file configuration."""
    train_csv: str = "./data/train.csv"
    test_csv: str = "./data/test.csv"
    id_column: str = "report_id"
    description_column: str = "description"
    diagnosis_column: str = "diagnosis"


@dataclass
class SpanClusteringConfig:
    """Step 0: Span clustering configuration."""
    k_min: int = 50
    k_max: int = 300
    k_step: int = 10
    max_reports: int = 100
    max_workers: int = 16
    batch_size: int = 256


@dataclass
class SlotInductionConfig:
    """Step 1: Slot induction configuration."""
    max_spans_per_cluster: int = 100
    max_workers: int = 16
    max_clusters: Optional[int] = None


@dataclass
class SlotMergingConfig:
    """Step 2: Slot merging configuration."""
    max_workers: int = 16
    max_families: Optional[int] = None


@dataclass
class SkeletonConfig:
    """Step 3: Skeleton building configuration."""
    num_sample_reports: int = 10


@dataclass
class SlotAssignmentConfig:
    """Step 4: Slot assignment configuration."""
    round1_max_workers: int = 16
    round2_max_workers: int = 16


@dataclass
class RefinementConfig:
    """Step 6: Template refinement configuration."""
    max_workers: int = 16


@dataclass
class InductionConfig:
    """Template induction pipeline configuration."""
    span_clustering: SpanClusteringConfig = field(default_factory=SpanClusteringConfig)
    slot_induction: SlotInductionConfig = field(default_factory=SlotInductionConfig)
    slot_merging: SlotMergingConfig = field(default_factory=SlotMergingConfig)
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    slot_assignment: SlotAssignmentConfig = field(default_factory=SlotAssignmentConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)


@dataclass
class StructuringConfig:
    """Report structuring configuration."""
    max_workers: int = 8
    include_omega: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    max_workers: int = 8
    num_reports: int = 50
    skip_bert: bool = False
    bert_models: Dict[str, str] = field(default_factory=lambda: {
        "macbert": "hfl/chinese-macbert-base",
        "roberta": "hfl/chinese-roberta-wwm-ext",
    })


@dataclass
class Config:
    """
    Root configuration object.

    Provides unified access to all ASTAR configuration settings.
    Supports loading from YAML files with environment variable overrides.
    """
    experiment_id: str = "exp_001"
    output_dir: str = "./output"

    llm: LLMConfig = field(default_factory=LLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    induction: InductionConfig = field(default_factory=InductionConfig)
    structuring: StructuringConfig = field(default_factory=StructuringConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def api_key(self) -> str:
        """Get API key from environment variable."""
        return os.getenv("ASTAR_API_KEY", "")

    @property
    def output_path(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.output_dir) / self.experiment_id

    def ensure_output_dir(self) -> Path:
        """Create and return the output directory."""
        path = self.output_path
        path.mkdir(parents=True, exist_ok=True)
        return path


def _load_nested_config(data: Dict[str, Any], config_class: type, prefix: str = "") -> Any:
    """Recursively load nested configuration dataclasses."""
    if data is None:
        return config_class()

    kwargs = {}
    for f in config_class.__dataclass_fields__.values():
        key = f.name
        if key in data:
            value = data[key]
            # Check if field type is a dataclass
            if hasattr(f.type, "__dataclass_fields__"):
                kwargs[key] = _load_nested_config(value, f.type, f"{prefix}{key}.")
            else:
                kwargs[key] = value

    return config_class(**kwargs)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml. If None, searches in default locations.

    Returns:
        Loaded Config object.
    """
    global _config

    # Find config file
    if config_path is None:
        search_paths = [
            Path.cwd() / "config.yaml",
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for path in search_paths:
            if path.exists():
                config_path = str(path)
                break

    # Load YAML if exists
    cfg_data = {}
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f) or {}

    # Build config object
    cfg = Config(
        experiment_id=cfg_data.get("experiment_id", "exp_001"),
        output_dir=cfg_data.get("output_dir", "./output"),
    )

    # Load nested configs
    if "llm" in cfg_data:
        cfg.llm = _load_nested_config(cfg_data["llm"], LLMConfig, "llm.")
    if "data" in cfg_data:
        cfg.data = _load_nested_config(cfg_data["data"], DataConfig, "data.")
    if "induction" in cfg_data:
        ind = cfg_data["induction"]
        cfg.induction = InductionConfig(
            span_clustering=_load_nested_config(
                ind.get("span_clustering"), SpanClusteringConfig
            ),
            slot_induction=_load_nested_config(
                ind.get("slot_induction"), SlotInductionConfig
            ),
            slot_merging=_load_nested_config(
                ind.get("slot_merging"), SlotMergingConfig
            ),
            skeleton=_load_nested_config(
                ind.get("skeleton"), SkeletonConfig
            ),
            slot_assignment=_load_nested_config(
                ind.get("slot_assignment"), SlotAssignmentConfig
            ),
            refinement=_load_nested_config(
                ind.get("refinement"), RefinementConfig
            ),
        )
    if "structuring" in cfg_data:
        cfg.structuring = _load_nested_config(
            cfg_data["structuring"], StructuringConfig
        )
    if "evaluation" in cfg_data:
        cfg.evaluation = _load_nested_config(
            cfg_data["evaluation"], EvaluationConfig
        )

    # Environment variable overrides
    if os.getenv("LLM_BASE_URL"):
        cfg.llm.base_url = os.getenv("LLM_BASE_URL")
    if os.getenv("LLM_MODEL"):
        cfg.llm.model = os.getenv("LLM_MODEL")

    _config = cfg
    return cfg


# Global singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration object.

    Loads config from default locations if not already loaded.

    Returns:
        Global Config instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
