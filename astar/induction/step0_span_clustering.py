"""
Step 0: Span Extraction and Clustering

Stage I - Part 1: Extract 7-tuple spans from reports, compute embeddings,
and perform K-means clustering.

This step extracts clinical concept spans from medical reports and groups
them into semantic clusters as preparation for slot induction.
"""

import random
from typing import List, Dict
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

from astar.core.config import get_config
from astar.core.llm import call_llm_json, get_embeddings
from astar.core.utils import save_jsonl

# System prompt for span extraction (Chinese medical domain)
SPAN_EXTRACTION_SYSTEM = """You are a medical report structuring expert. Extract all clinical concept spans from the given medical report.

Each span is a 7-tuple:
{
  "spans": [
    {
      "a_struct": "anatomical structure (e.g., fetus, uterus, placenta)",
      "a_sub": "sub-structure (e.g., skull, spine, empty if none)",
      "a_attr": "attribute name (e.g., biparietal diameter, morphology)",
      "a_type": "value type (categorical/numerical/freetext)",
      "a_val": "normalized value (use <NUM> for numbers, e.g., <NUM>mm)",
      "a_role": "clinical role (finding/measurement/diagnosis)",
      "a_sec": "report section (desc/conclusion)"
    }
  ]
}"""


def extract_spans_from_report(report: Dict[str, str], report_id: int) -> List[Dict]:
    """
    Extract 7-tuple spans from a single report.

    Args:
        report: Report dictionary with description and diagnosis fields.
        report_id: Unique identifier for the report.

    Returns:
        List of span dictionaries.
    """
    cfg = get_config()
    desc = report.get(cfg.data.description_column, "") or ""
    diag = report.get(cfg.data.diagnosis_column, "") or ""

    user_prompt = f"Please extract all clinical concept spans from the following report:\n\nDescription:\n{desc}\n\nDiagnosis:\n{diag}"
    result = call_llm_json(SPAN_EXTRACTION_SYSTEM, user_prompt)

    spans = []
    if result and "spans" in result:
        for s in result["spans"]:
            spans.append({
                "a_struct": s.get("a_struct", ""),
                "a_sub": s.get("a_sub", ""),
                "a_attr": s.get("a_attr", ""),
                "a_type": s.get("a_type", "freetext"),
                "a_val": s.get("a_val", ""),
                "a_role": s.get("a_role", "finding"),
                "a_sec": s.get("a_sec", "desc"),
                "report_id": report_id,
            })
    return spans


def deduplicate_spans(spans: List[Dict]) -> List[Dict]:
    """Remove duplicate spans based on (a_val, a_role) key."""
    seen = set()
    unique = []
    for s in spans:
        key = (s.get("a_val", ""), s.get("a_role", ""))
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def extract_all_spans(df: pd.DataFrame, max_reports: int, max_workers: int) -> List[Dict]:
    """
    Extract spans from all reports in parallel.

    Args:
        df: DataFrame containing reports.
        max_reports: Maximum number of reports to process.
        max_workers: Number of parallel workers.

    Returns:
        List of deduplicated spans.
    """
    reports = [(idx, row.to_dict()) for idx, row in df.iterrows() if idx < max_reports]
    all_spans = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(extract_spans_from_report, r, idx): idx
            for idx, r in reports
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting spans"):
            try:
                all_spans.extend(fut.result())
            except Exception as e:
                print(f"[Step0] Error: {e}")

    return deduplicate_spans(all_spans)


def build_span_text(span: Dict) -> str:
    """Build text representation for embedding."""
    parts = [
        span.get("a_struct", ""),
        span.get("a_sub", ""),
        span.get("a_attr", ""),
        span.get("a_val", ""),
    ]
    return " ".join(p for p in parts if p)


def embed_spans(spans: List[Dict], batch_size: int, max_workers: int) -> np.ndarray:
    """
    Compute embeddings for all spans.

    Args:
        spans: List of span dictionaries.
        batch_size: Batch size for embedding API calls.
        max_workers: Number of parallel workers.

    Returns:
        NumPy array of embeddings.
    """
    texts = [build_span_text(s) for s in spans]

    # Get embeddings using unified function (handles batching internally)
    embeddings = get_embeddings(texts, batch_size=batch_size)

    return np.array(embeddings)


def find_optimal_k(embeddings: np.ndarray, k_min: int, k_max: int, step: int) -> int:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Array of embedding vectors.
        k_min: Minimum k to try.
        k_max: Maximum k to try.
        step: Step size for k search.

    Returns:
        Optimal k value.
    """
    n_samples = len(embeddings)

    # Ensure k range is valid: k must be >= 2 and < n_samples
    k_min = max(2, min(k_min, n_samples - 1))
    k_max = min(k_max, n_samples - 1)

    if k_min >= k_max:
        print(f"[Step0] Too few samples ({n_samples}), using K={k_min}")
        return k_min

    best_k, best_score = k_min, -1
    for k in tqdm(range(k_min, k_max + 1, step), desc="Finding optimal K"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k

    print(f"[Step0] Optimal K={best_k} (silhouette={best_score:.4f})")
    return best_k


def run_step0(output_dir=None):
    """
    Execute Step 0: Span Extraction and Clustering.

    Extracts spans from reports, computes embeddings, and clusters them.
    Saves results to spans_with_cluster.jsonl.
    """
    cfg = get_config()
    if output_dir is None:
        output_dir = cfg.ensure_output_dir()

    # Load data
    df = pd.read_csv(cfg.data.train_csv, encoding="utf-8")
    df.columns = df.columns.str.strip()
    print(f"[Step0] Loaded {len(df)} reports")

    sc_cfg = cfg.induction.span_clustering

    # Extract spans
    spans = extract_all_spans(df, sc_cfg.max_reports, sc_cfg.max_workers)
    if not spans:
        print("[Step0] No spans extracted")
        return []

    print(f"[Step0] Extracted {len(spans)} unique spans")

    # Compute embeddings
    embeddings = embed_spans(spans, sc_cfg.batch_size, sc_cfg.max_workers)

    # Find optimal K and cluster
    k = find_optimal_k(embeddings, sc_cfg.k_min, sc_cfg.k_max, sc_cfg.k_step)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Add cluster labels
    for i, s in enumerate(spans):
        s["cluster_id"] = int(labels[i])

    # Save results
    save_jsonl(spans, output_dir / "spans_with_cluster.jsonl")
    print(f"[Step0] Done. {len(spans)} spans -> {k} clusters")

    return spans


if __name__ == "__main__":
    run_step0()
