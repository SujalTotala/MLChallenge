"""
Hard negative sampling and training dataset construction module.
Prioritizes non-trivial, challenging negative pairs to maximize classifier precision.
"""

from typing import Dict, List, Set, Tuple, Any
import numpy as np
import pandas as pd
from src.utils import logger, time_block


def sample_hard_negatives(
    s1_id: str,
    candidates: List[Tuple[str, float, int]],
    ground_truth_set: Set[str],
    max_negatives_per_positive: int = 5,
    max_total_negatives: int = 15
) -> Tuple[List[str], List[str]]:
    """
    Select positive target IDs and prioritized hard negative target IDs for a single S1 entity.
    Returns: (positive_ids, negative_ids)
    """
    positives = []
    negatives_scored = []

    for cand_id, prelim_score, channel_support in candidates:
        if cand_id in ground_truth_set:
            positives.append(cand_id)
        else:
            # Score hardness: higher prelim score + higher channel support = harder negative!
            hardness_weight = prelim_score + (channel_support * 1.5)
            negatives_scored.append((cand_id, hardness_weight))

    # Sort negatives by hardness (descending)
    negatives_scored.sort(key=lambda x: x[1], reverse=True)

    # Determine negative quota
    if len(positives) > 0:
        quota = min(len(positives) * max_negatives_per_positive, max_total_negatives)
    else:
        # Singleton entity with no ground truth matches
        quota = min(max_negatives_per_positive, len(negatives_scored))

    selected_negatives = [n[0] for n in negatives_scored[:quota]]
    return positives, selected_negatives
