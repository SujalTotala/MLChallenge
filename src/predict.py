"""
Inference and submission generation pipeline for Business Entity Resolution.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import joblib
import numpy as np
import pandas as pd

from src.config import (
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    OUTPUT_MATCHING_PATH,
    OUTPUT_CANDIDATES_PATH,
    MODEL_PATH,
    METADATA_PATH,
    DEFAULT_DECISION_THRESHOLD,
)
from src.data_loader import load_test_data, preprocess_source_df
from src.candidate_generation import generate_candidates_for_dataset, export_candidates_file
from src.features import extract_features_for_candidates
from src.utils import save_submission_tsv, logger, time_block


def get_all_test_s1_ids() -> List[str]:
    """Fast scan to get complete list of required S1 test entity IDs."""
    with open(TEST_SOURCE1_PATH, "r", encoding="utf-8") as f:
        next(f, None)
        return [line.split("\t", 1)[0].strip() for line in f if line.strip()]


@time_block("Test Prediction and Submission Generation")
def run_predict_pipeline(
    nrows: Optional[int] = None,
    output_matching_path: Path = OUTPUT_MATCHING_PATH,
    output_candidates_path: Path = OUTPUT_CANDIDATES_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    threshold: Optional[float] = None
):
    """
    Execute full inference pipeline on the test dataset.
    """
    # 1. Load test data
    s1_raw, s2_raw, s3_raw = load_test_data(nrows=nrows)
    # Get all required test IDs to ensure submission completeness
    all_required_s1_ids = get_all_test_s1_ids()
    logger.info(f"Loaded {len(s1_raw):,} Test Source 1 entities to score (Total required S1 in test: {len(all_required_s1_ids):,})")

    # 2. Preprocess records
    logger.info("Preprocessing test records...")
    s1_prep = preprocess_source_df(s1_raw)
    s2_prep = preprocess_source_df(s2_raw)
    s3_prep = preprocess_source_df(s3_raw)
    targets_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

    # 3. Candidate Generation (Blocking)
    logger.info("Generating candidates on test set...")
    candidates_dict = generate_candidates_for_dataset(
        s1_preprocessed=s1_prep,
        s2_preprocessed=s2_prep,
        s3_preprocessed=s3_prep,
    )

    # 4. Export candidate pairs file
    logger.info(f"Exporting candidate pairs to {output_candidates_path}...")
    export_candidates_file(
        candidates_dict=candidates_dict,
        all_s1_ids=all_required_s1_ids,
        output_path=output_candidates_path
    )

    # 5. Extract features
    logger.info("Extracting test pairwise features...")
    X_test, test_pair_ids = extract_features_for_candidates(
        candidates_dict=candidates_dict,
        s1_preprocessed=s1_prep,
        targets_preprocessed=targets_prep
    )

    # 6. Load model and metadata
    logger.info(f"Loading trained matcher from {model_path}...")
    model = joblib.load(model_path)

    if threshold is None:
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                threshold = meta.get("optimal_threshold", DEFAULT_DECISION_THRESHOLD)
        else:
            threshold = DEFAULT_DECISION_THRESHOLD
    
    logger.info(f"Applying decision threshold: {threshold:.2f}")

    # 7. Predict and filter matches
    if len(X_test) > 0:
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = np.zeros(0)

    predictions_map: Dict[str, Set[str]] = {s1: set() for s1 in all_required_s1_ids}
    for (s1, cand), prob in zip(test_pair_ids, probs):
        if prob >= threshold:
            predictions_map[s1].add(cand)

    # 8. Export matching results
    logger.info(f"Exporting final matches to {output_matching_path}...")
    save_submission_tsv(
        output_path=output_matching_path,
        mapping=predictions_map,
        all_s1_ids=all_required_s1_ids,
        id_col_name="matched_entity_ids"
    )

    # Print summary statistics
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
