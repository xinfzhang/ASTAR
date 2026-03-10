"""
Unified File I/O Utilities

This module provides a single implementation of file I/O operations for all ASTAR components.

Usage:
    from astar.core.utils import load_json, save_json, load_jsonl, save_jsonl

    data = load_json(Path("data.json"))
    save_json(data, Path("output.json"))

    records = load_jsonl(Path("records.jsonl"))
    save_jsonl(records, Path("output.jsonl"))
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, creating it if necessary.

    Args:
        path: Directory path (can be file path, will use parent).

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    if path.suffix:
        # It's a file path, use parent directory
        path = path.parent
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Union[str, Path]) -> Any:
    """
    Load JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON content.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save object as JSON file.

    Args:
        obj: Object to serialize.
        path: Output file path.
        indent: JSON indentation level.
    """
    path = Path(path)
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    print(f"[IO] Saved: {path}")


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load JSONL (JSON Lines) file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(items: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save list of objects as JSONL file.

    Args:
        items: List of objects to serialize.
        path: Output file path.
    """
    path = Path(path)
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[IO] Saved: {path}")


def load_csv_as_records(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load CSV file as list of record dictionaries.

    Args:
        path: Path to CSV file.

    Returns:
        List of row dictionaries.
    """
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8")
    df.columns = df.columns.str.strip()
    return df.to_dict(orient="records")


def json_dumps(obj: Any, **kwargs) -> str:
    """
    Serialize object to JSON string with default settings.

    Args:
        obj: Object to serialize.
        **kwargs: Additional json.dumps arguments.

    Returns:
        JSON string.
    """
    defaults = {"ensure_ascii": False, "indent": 2}
    defaults.update(kwargs)
    return json.dumps(obj, **defaults)
