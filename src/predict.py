"""
Inference and submission generation pipeline for Business Entity Resolution.
Applies exact candidate blocking, LightGBM matcher scoring, singleton-aware decision rules,
and writes official submission TSV files.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict

from src.config import (
    TEST_SOURCE1_PATH,
    OUTPUT_MATCHING_PATH,
    OUTPUT_CANDIDATES_PATH,
    MODEL_PATH,
    METADATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
    MARGIN_THRESHOLD,
)
from src.data_loader import load_test_data, preprocess_source_df
from src.candidate_generation import generate_candidates_for_dataset, export_candidates_file
from src.features import extract_features_for_candidates
from src.utils import save_submission_tsv, logger, time_block


def get_all_test_s1_ids() -> List[str]:
    """Fast scan to retrieve exact ordered list of required test S1 entity IDs."""
    with open(TEST_SOURCE1_PATH, "r", encoding="utf-8") as f:
        next(f, None)
        return [line.split("\t", 1)[0].strip() for line in f if line.strip()]


@time_block("Test Inference & Output TSV Generation")
def run_predict_pipeline(
    nrows: Optional[int] = None,
    output_matching_path: Path = OUTPUT_MATCHING_PATH,
    output_candidates_path: Path = OUTPUT_CANDIDATES_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    threshold: Optional[float] = None,
    margin: Optional[float] = None
):
    """
    Execute full inference pipeline on the test dataset.
    ALWAYS ensures every required test S1 entity is present in both output TSV files.
    """
    # 1. Load Test Data
    s1_raw, s2_raw, s3_raw = load_test_data(nrows=nrows)
    # Always load full required test S1 IDs for validator compliance
    all_required_s1_ids = get_all_test_s1_ids()
    logger.info(f"Loaded {len(s1_raw):,} Test Source 1 entities to score (Total required S1 entities: {len(all_required_s1_ids):,})")

    # 2. Preprocess Records
    logger.info("Preprocessing test entity records...")
    s1_prep = preprocess_source_df(s1_raw)
    s2_prep = preprocess_source_df(s2_raw)
    s3_prep = preprocess_source_df(s3_raw)
    targets_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    # 3. Candidate Generation (Multi-Channel Blocking)
    logger.info("Generating candidate pairs on test set...")
    candidates_dict = generate_candidates_for_dataset(
        s1_preprocessed=s1_prep,
        s2_preprocessed=s2_prep,
        s3_preprocessed=s3_prep,
    )

    # 4. Export Candidate Pairs File (candidate_pairs.tsv)
    logger.info(f"Exporting candidate_pairs.tsv to {output_candidates_path}...")
    export_candidates_file(
        candidates_dict=candidates_dict,
        all_s1_ids=all_required_s1_ids,
        output_path=output_candidates_path
    )

    # 5. Extract Pairwise Features
    logger.info("Extracting test pairwise features...")
    X_test, test_pair_ids = extract_features_for_candidates(
        candidates_dict=candidates_dict,
        s1_preprocessed=s1_prep,
        targets_preprocessed=targets_prep
    )

    # 6. Load Trained Matcher Model & Decision Parameters
    logger.info(f"Loading trained matcher model from {model_path}...")
    model = joblib.load(model_path)

    if threshold is None or margin is None:
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                if threshold is None:
                    threshold = meta.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD)
                if margin is None:
                    margin = meta.get("optimal_margin", MARGIN_THRESHOLD)
        else:
            threshold = threshold or DEFAULT_DECISION_THRESHOLD
            margin = margin or MARGIN_THRESHOLD

    logger.info(f"Applying Singleton-Aware Decision Rules: Threshold={threshold:.2f}, Margin={margin:.2f}")

    # 7. Model Inference & Singleton-Aware Filtering
    if len(X_test) > 0:
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = np.zeros(0)

    # Group predicted probabilities per S1 entity
    s1_preds = defaultdict(list)
    for (s1, cand), prob in zip(test_pair_ids, probs):
        s1_preds[s1].append((cand, prob))

    predictions_map: Dict[str, Set[str]] = {s1: set() for s1 in all_required_s1_ids}

    for s1, pairs in s1_preds.items():
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[1], reverse=True)
        best_cand, best_prob = pairs[0]

        if best_prob >= threshold:
            second_prob = pairs[1][1] if len(pairs) > 1 else 0.0
            if (best_prob - second_prob) >= margin or best_prob >= (threshold + 0.05):
                predictions_map[s1].add(best_cand)

                # Support multiple matches if candidate is very close to top prob
                for cand, prob in pairs[1:]:
                    if prob >= threshold and (best_prob - prob) <= 0.03:
                        predictions_map[s1].add(cand)

    # 8. Export Matching Results File (matching_results.tsv)
    logger.info(f"Exporting matching_results.tsv to {output_matching_path}...")
    save_submission_tsv(
        output_path=output_matching_path,
        mapping=predictions_map,
        all_s1_ids=all_required_s1_ids,
        id_col_name="matched_entity_ids"
    )

    # Inference Summary
    match_counts = [len(m) for m in predictions_map.values()]
    singletons = sum(1 for c in match_counts if c == 0)
    matched_s1 = len(all_required_s1_ids) - singletons
    total_links = sum(match_counts)

    logger.info("=" * 60)
    logger.info("TEST INFERENCE SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Total S1 Entities:     {len(all_required_s1_ids):,}")
    logger.info(f"  Predicted Singletons:  {singletons:,} ({singletons/len(all_required_s1_ids):.2%})")
    logger.info(f"  Matched S1 Entities:   {matched_s1:,} ({matched_s1/len(all_required_s1_ids):.2%})")
    logger.info(f"  Total Links Predicted: {total_links:,}")
    logger.info(f"  Avg Matches / S1:      {np.mean(match_counts):.2f}")
    logger.info("=" * 60)

    return predictions_map


if __name__ == "__main__":
    run_predict_pipeline()
