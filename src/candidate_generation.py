"""
Candidate generation pipeline for Business Entity Resolution.
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.config import BLOCKING_CONFIG, OUTPUT_CANDIDATES_PATH
from src.blocking import BlockingEngine
from src.data_loader import preprocess_source_df
from src.utils import logger, time_block, save_submission_tsv, parse_id_list


@time_block("Candidate Generation")
def generate_candidates_for_dataset(
    s1_preprocessed: pd.DataFrame,
    s2_preprocessed: pd.DataFrame,
    s3_preprocessed: pd.DataFrame,
    max_candidates: int = BLOCKING_CONFIG["max_candidates_per_s1"],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Generate candidate matches for all Source 1 records against Source 2 and Source 3.
    Returns: mapping s1_id -> list of (candidate_id, preliminary_score)
    """
    # Combine Source 2 and Source 3 target records
    logger.info("Combining Source 2 and Source 3 target databases...")
    targets_df = pd.concat([s2_preprocessed, s3_preprocessed], ignore_index=True)
    
    logger.info(f"Target database size: {len(targets_df):,} records (S2: {len(s2_preprocessed):,}, S3: {len(s3_preprocessed):,})")
    
    blocking_engine = BlockingEngine(targets_df)
    
    logger.info(f"Generating candidates for {len(s1_preprocessed):,} Source 1 records...")
    candidates_dict = {}
    
    # Progress bar for candidate retrieval
    for row in tqdm(s1_preprocessed.itertuples(index=False), total=len(s1_preprocessed), desc="Candidate Blocking"):
        s1_id = row.entity_id
        cands = blocking_engine.get_candidates_for_record(row, max_candidates=max_candidates)
        candidates_dict[s1_id] = cands

    return candidates_dict


def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[Tuple[str, float]]],
    ground_truth_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Evaluate candidate recall against training ground truth and print statistics.
    """
    logger.info("Evaluating candidate blocking recall...")
    total_true_matches = 0
    recovered_matches = 0
    s1_with_all_recovered = 0
    total_s1_with_matches = 0

    candidate_counts = []

    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        true_matches = parse_id_list(row.get("matched_entity_ids"))
        
        cands_with_scores = candidates_dict.get(s1_id, [])
        candidate_ids = {c[0] for c in cands_with_scores}
        candidate_counts.append(len(candidate_ids))

        if len(true_matches) > 0:
            total_s1_with_matches += 1
            total_true_matches += len(true_matches)
            overlap = true_matches & candidate_ids
            recovered_matches += len(overlap)
            if overlap == true_matches:
                s1_with_all_recovered += 1

    recall = (recovered_matches / total_true_matches) if total_true_matches > 0 else 0.0
    perfect_s1_ratio = (s1_with_all_recovered / total_s1_with_matches) if total_s1_with_matches > 0 else 0.0

    stats = {
        "candidate_recall": recall,
        "total_true_links": total_true_matches,
        "recovered_links": recovered_matches,
        "full_recall_s1_ratio": perfect_s1_ratio,
        "mean_candidates_per_s1": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "median_candidates_per_s1": float(np.median(candidate_counts)) if candidate_counts else 0.0,
        "p95_candidates_per_s1": float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
        "max_candidates_per_s1": int(max(candidate_counts)) if candidate_counts else 0,
    }

    logger.info(f"--- CANDIDATE BLOCKING METRICS ---")
    logger.info(f"Candidate Recall: {stats['candidate_recall']:.4f} ({recovered_matches:,} / {total_true_matches:,} true links recovered)")
    logger.info(f"S1 with 100% Candidates Found: {stats['full_recall_s1_ratio']:.4f}")
    logger.info(f"Mean Candidates per S1: {stats['mean_candidates_per_s1']:.2f}")
    logger.info(f"Median Candidates per S1: {stats['median_candidates_per_s1']:.1f}")
    logger.info(f"95th percentile Candidates per S1: {stats['p95_candidates_per_s1']:.1f}")
    logger.info(f"Max Candidates per S1: {stats['max_candidates_per_s1']}")

    return stats


def export_candidates_file(
    candidates_dict: Dict[str, List[Tuple[str, float]]],
    all_s1_ids: List[str],
    output_path: Path = OUTPUT_CANDIDATES_PATH
):
    """
    Save candidate pairs in TSV format conforming to the challenge submission schema.
    """
    set_dict = {s1: {c[0] for c in cands} for s1, cands in candidates_dict.items()}
    save_submission_tsv(
        output_path=output_path,
        mapping=set_dict,
        all_s1_ids=all_s1_ids,
        id_col_name="candidate_entity_ids"
    )
