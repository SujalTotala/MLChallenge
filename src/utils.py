"""
Utility functions for evaluation, logging, timing, and formatting.
"""

import time
import logging
from functools import wraps
from pathlib import Path
from typing import Dict, Set, List, Tuple, Any, Optional, Union
import pandas as pd
import numpy as np

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("ER_Pipeline")


def time_block(name: str):
    """Context manager / decorator for execution timing."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"Starting {name}...")
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"Finished {name} in {elapsed:.2f}s")
            return result
        return wrapper
    return decorator


def parse_ground_truth_file(gt_input: Union[Path, pd.DataFrame, str], nrows: Optional[int] = None) -> Dict[str, Set[str]]:
    """
    Parse ground truth into a dictionary mapping:
    source1_entity_id -> set of matched_entity_ids
    Accepts either DataFrame or Path/str.
    """
    if isinstance(gt_input, pd.DataFrame):
        gt_df = gt_input
    else:
        gt_df = pd.read_csv(gt_input, sep="\t", dtype=str, nrows=nrows)
    
    gt_dict = {}
    s1_col = "source1_entity_id" if "source1_entity_id" in gt_df.columns else gt_df.columns[0]
    match_col = "matched_entity_ids" if "matched_entity_ids" in gt_df.columns else gt_df.columns[1]

    s1_vals = gt_df[s1_col].astype(str).str.strip().tolist()
    match_vals = gt_df[match_col].fillna("").astype(str).tolist()

    for s1, matches_str in zip(s1_vals, match_vals):
        if not matches_str or matches_str in {"nan", "none", "<na>"}:
            gt_dict[s1] = set()
        else:
            gt_dict[s1] = {m.strip() for m in matches_str.split(",") if m.strip()}
    return gt_dict


def parse_id_list(value: Any) -> Set[str]:
    """Parse comma-separated ID string into a set of clean IDs."""
    if pd.isna(value) or not str(value).strip():
        return set()
    return {x.strip() for x in str(value).split(",") if x.strip()}


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F0.5 for a single Source 1 entity.
    
    Rules for singletons:
    - If truth is empty and pred is empty: Perfect prediction -> P=1.0, R=1.0, F0.5=1.0
    - If truth is empty and pred is non-empty: False positive match -> P=0.0, R=1.0, F0.5=0.0
    - If truth is non-empty and pred is empty: Missed match -> P=0.0, R=0.0, F0.5=0.0
    - If both non-empty: Standard precision, recall, and F0.5 formula
    """
    is_true_empty = len(true_matches) == 0
    is_pred_empty = len(pred_matches) == 0

    if is_true_empty and is_pred_empty:
        return 1.0, 1.0, 1.0
    elif is_true_empty and not is_pred_empty:
        return 0.0, 1.0, 0.0
    elif not is_true_empty and is_pred_empty:
        return 0.0, 0.0, 0.0

    tp = len(true_matches & pred_matches)
    precision = tp / len(pred_matches) if len(pred_matches) > 0 else 0.0
    recall = tp / len(true_matches) if len(true_matches) > 0 else 0.0

    denom = 0.25 * precision + recall
    if denom > 0:
        f05 = (1.25 * precision * recall) / denom
    else:
        f05 = 0.0

    return precision, recall, f05


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute macro-averaged F0.5, Precision, and Recall across all specified Source 1 entities.
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    f05_scores = []
    precision_scores = []
    recall_scores = []
    
    singletons_total = 0
    singletons_correct = 0
    non_singletons_total = 0

    for s1 in s1_ids:
        true_set = ground_truth.get(s1, set())
        pred_set = predictions.get(s1, set())

        p, r, f = compute_entity_f05(true_set, pred_set)
        f05_scores.append(f)
        precision_scores.append(p)
        recall_scores.append(r)

        if len(true_set) == 0:
            singletons_total += 1
            if len(pred_set) == 0:
                singletons_correct += 1
        else:
            non_singletons_total += 1

    macro_f05 = float(np.mean(f05_scores)) if f05_scores else 0.0
    macro_p = float(np.mean(precision_scores)) if precision_scores else 0.0
    macro_r = float(np.mean(recall_scores)) if recall_scores else 0.0
    singleton_acc = (singletons_correct / singletons_total) if singletons_total > 0 else 1.0

    return {
        "macro_f05": macro_f05,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "singletons_total": singletons_total,
        "singletons_correct": singletons_correct,
        "singleton_accuracy": singleton_acc,
        "non_singletons_total": non_singletons_total,
        "total_evaluated_s1": len(s1_ids),
    }


def save_submission_tsv(
    output_path: Path,
    mapping: Dict[str, Set[str]],
    all_s1_ids: List[str],
    id_col_name: str = "matched_entity_ids"
):
    """
    Write results to a TSV compliant with official validator:
    - Tab-separated
    - Header: source1_entity_id \t matched_entity_ids (or candidate_entity_ids)
    - Comma-separated match IDs in second column
    - Every S1 ID appears exactly once
    - Empty string for singletons
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{id_col_name}\n")
        for s1 in all_s1_ids:
            matches = sorted(list(mapping.get(s1, set())))
            match_str = ",".join(matches)
            f.write(f"{s1}\t{match_str}\n")
    logger.info(f"Saved {len(all_s1_ids)} records to {output_path}")
