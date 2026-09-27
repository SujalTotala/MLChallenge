"""
Candidate generation and blocking recall evaluation module.
Generates candidate_pairs.tsv and blocking performance reports.
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import TOP_K_CANDIDATES, OUTPUT_CANDIDATES_PATH, BLOCKING_REPORT_PATH
from src.blocking import MultiChannelBlockingEngine
from src.utils import logger, time_block, save_submission_tsv, parse_id_list


@time_block("Candidate Generation (Multi-Channel Blocking)")
def generate_candidates_for_dataset(
    s1_preprocessed: pd.DataFrame,
    s2_preprocessed: pd.DataFrame,
    s3_preprocessed: pd.DataFrame,
    top_k: int = TOP_K_CANDIDATES,
) -> Dict[str, List[Tuple[str, float, int]]]:
    """
    Generate candidate matches for all Source 1 records against Source 2 and Source 3.
    Returns: mapping s1_id -> list of (candidate_id, preliminary_score, channel_support_count)
    """
    logger.info("Combining Source 2 and Source 3 target databases...")
    targets_df = pd.concat([s2_preprocessed, s3_preprocessed], ignore_index=True)
    logger.info(f"Target database size: {len(targets_df):,} records (S2: {len(s2_preprocessed):,}, S3: {len(s3_preprocessed):,})")

    blocking_engine = MultiChannelBlockingEngine(targets_df)

    logger.info(f"Generating candidates for {len(s1_preprocessed):,} Source 1 records (top_k={top_k})...")
    candidates_dict = {}

    for row in tqdm(s1_preprocessed.itertuples(index=False), total=len(s1_preprocessed), desc="Blocking Channels"):
        s1_id = row.entity_id
        cands = blocking_engine.retrieve_candidates_for_s1(row, top_k=top_k)
        candidates_dict[s1_id] = cands

    return candidates_dict


def evaluate_blocking_recall(
    candidates_dict: Dict[str, List[Tuple[str, float, int]]],
    ground_truth_df: pd.DataFrame,
    s1_preprocessed: Optional[pd.DataFrame] = None,
    total_target_records: int = 10000000
) -> Dict[str, Any]:
    """
    Evaluate blocking metrics against ground truth and write reports/blocking_report.txt:
    - pair candidate recall
    - entity-complete candidate recall
    - average candidates per S1
    - median candidates per S1
    - P95 candidates per S1
    - maximum candidates per S1
    - candidate reduction ratio
    - recall breakdown by country, multiple matches, and singletons.
    """
    logger.info("Evaluating multi-channel candidate blocking recall...")

    total_true_matches = 0
    recovered_matches = 0
    s1_with_all_recovered = 0
    total_s1_with_matches = 0
    singletons_count = 0

    candidate_counts = []
    
    # Country-level tracking
    s1_country_map = {}
    if s1_preprocessed is not None and "country" in s1_preprocessed.columns:
        s1_country_map = dict(zip(s1_preprocessed["entity_id"], s1_preprocessed["country"]))

    country_true = defaultdict(int)
    country_recovered = defaultdict(int)

    for _, row in ground_truth_df.iterrows():
        s1_id = str(row["source1_entity_id"]).strip()
        true_matches = parse_id_list(row.get("matched_entity_ids"))
        cands_info = candidates_dict.get(s1_id, [])
        cand_ids = {c[0] for c in cands_info}
        
        candidate_counts.append(len(cand_ids))
        country = s1_country_map.get(s1_id, "unknown")

        if len(true_matches) == 0:
            singletons_count += 1
        else:
            total_s1_with_matches += 1
            total_true_matches += len(true_matches)
            country_true[country] += len(true_matches)

            overlap = true_matches & cand_ids
            recovered_matches += len(overlap)
            country_recovered[country] += len(overlap)

            if overlap == true_matches:
                s1_with_all_recovered += 1

    pair_recall = (recovered_matches / total_true_matches) if total_true_matches > 0 else 0.0
    entity_complete_recall = (s1_with_all_recovered / total_s1_with_matches) if total_s1_with_matches > 0 else 0.0

    mean_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    median_cands = float(np.median(candidate_counts)) if candidate_counts else 0.0
    p95_cands = float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0
    max_cands = int(max(candidate_counts)) if candidate_counts else 0

    # Reduction ratio: 1 - (total candidates evaluated / (N_s1 * N_targets))
    total_candidates_generated = sum(candidate_counts)
    total_possible_pairs = max(1, len(candidate_counts) * total_target_records)
    reduction_ratio = 1.0 - (total_candidates_generated / total_possible_pairs)

    country_recall = {}
    for cntry, t_count in country_true.items():
        rec = (country_recovered[cntry] / t_count) if t_count > 0 else 0.0
        country_recall[cntry] = rec

    stats = {
        "raw_candidate_count": total_candidates_generated,
        "final_candidate_count": total_candidates_generated,
        "pair_candidate_recall": pair_recall,
        "entity_complete_recall": entity_complete_recall,
        "mean_candidates_per_s1": mean_cands,
        "median_candidates_per_s1": median_cands,
        "p95_candidates_per_s1": p95_cands,
        "max_candidates_per_s1": max_cands,
        "candidate_reduction_ratio": reduction_ratio,
        "country_recall": country_recall,
        "total_true_links": total_true_matches,
        "recovered_links": recovered_matches,
        "singletons_count": singletons_count,
        "non_singletons_count": total_s1_with_matches,
    }

    # Print summary
    logger.info("=" * 60)
    logger.info("BLOCKING EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"  ★ Pair Candidate Recall:           {pair_recall:.4f} ({recovered_matches:,} / {total_true_matches:,} links)")
    logger.info(f"  ★ Entity-Complete Recall:          {entity_complete_recall:.4f} ({s1_with_all_recovered:,} / {total_s1_with_matches:,} S1)")
    logger.info(f"  Average Candidates per S1:         {mean_cands:.2f}")
    logger.info(f"  Median Candidates per S1:          {median_cands:.1f}")
    logger.info(f"  P95 Candidates per S1:             {p95_cands:.1f}")
    logger.info(f"  Max Candidates per S1:             {max_cands}")
    logger.info(f"  Candidate Reduction Ratio:         {reduction_ratio:.8f}")
    if country_recall:
        logger.info("  Recall by Country:")
        for cntry, rec in country_recall.items():
            logger.info(f"    - {cntry.upper()}: Recall = {rec:.4f}")
    logger.info("=" * 60)

    # Write reports/blocking_report.txt
    BLOCKING_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BLOCKING_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("OFFICIAL BLOCKING & CANDIDATE GENERATION REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Raw Candidate Count:          {total_candidates_generated:,}\n")
        f.write(f"Final Candidate Count:        {total_candidates_generated:,}\n")
        f.write(f"Average Candidates/S1:        {mean_cands:.2f}\n")
        f.write(f"Median Candidates/S1:         {median_cands:.1f}\n")
        f.write(f"P95 Candidates/S1:            {p95_cands:.1f}\n")
        f.write(f"Max Candidates/S1:            {max_cands}\n")
        f.write(f"Pair Candidate Recall:        {pair_recall:.4f}\n")
        f.write(f"Entity-Complete Recall:       {entity_complete_recall:.4f}\n")
        f.write(f"Candidate Reduction Ratio:    {reduction_ratio:.8f}\n\n")
        f.write("Country Breakdown:\n")
        for cntry, rec in country_recall.items():
            f.write(f"  - {cntry.upper()}: Recall = {rec:.4f}\n")
    logger.info(f"Saved blocking report to {BLOCKING_REPORT_PATH}")

    return stats


def export_candidates_file(
    candidates_dict: Dict[str, List[Tuple[str, float, int]]],
    all_s1_ids: List[str],
    output_path: Path = OUTPUT_CANDIDATES_PATH
):
    """
    Save candidate_pairs.tsv complying strictly with official challenge schema:
    source1_entity_id \t candidate_entity_ids
    """
    set_dict = {s1: {c[0] for c in cands} for s1, cands in candidates_dict.items()}
    save_submission_tsv(
        output_path=output_path,
        mapping=set_dict,
        all_s1_ids=all_s1_ids,
        id_col_name="candidate_entity_ids"
    )
