#!/usr/bin/env python3
"""
End-to-End Execution Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import (
    TRAIN_SOURCE1_PATH,
    OUTPUT_MATCHING_PATH,
    OUTPUT_CANDIDATES_PATH,
    TEST_DIR,
)
from src.data_loader import load_train_data, preprocess_source_df
from src.candidate_generation import generate_candidates_for_dataset, evaluate_blocking_recall
from src.train_model import train_matcher_model
from src.predict import run_predict_pipeline
from src.eda import run_eda
from src.utils import parse_ground_truth_file, logger, time_block
import pandas as pd


def run_validator() -> bool:
    """Run the official validator script on the generated outputs."""
    validator_path = Path("utils/validate_submission.py")
    if not validator_path.exists():
        logger.warning(f"Validator script not found at {validator_path}. Skipping.")
        return True

    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(OUTPUT_MATCHING_PATH),
        "--candidate", str(OUTPUT_CANDIDATES_PATH),
        "--test-dir", str(TEST_DIR),
    ]
    logger.info(f"Running validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


@time_block("Complete Entity Resolution Pipeline")
def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "eda", "train", "predict", "validate"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25000,
        help="Number of records to sample for fast training/evaluation (use 0 for full dataset, default: 25000)"
    )
    parser.add_argument(
        "--test-sample-size",
        type=int,
        default=25000,
        help="Number of test records to score during test stage (use 0 for full test set, default: 25000)"
    )
    parser.add_argument(
        "--model-type",
        choices=["hist_gb", "rf"],
        default="hist_gb",
        help="Classifier model architecture (default: hist_gb)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold for prediction"
    )
    args = parser.parse_args()

    nrows = args.sample_size if args.sample_size > 0 else None
    test_nrows = args.test_sample_size if args.test_sample_size > 0 else None

    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    logger.info(f"Stage: {args.stage} | Train Sample Size: {nrows or 'FULL'} | Test Score Size: {test_nrows or 'FULL'} | Model: {args.model_type}")
    logger.info("=" * 60)

    # Stage: EDA
    if args.stage in ["all", "eda"]:
        logger.info("\n>>> STAGE 1: EXPLORATORY DATA ANALYSIS (EDA)")
        run_eda(sample_size=nrows or 50000)
        if args.stage == "eda":
            return

    # Stage: Train Model
    if args.stage in ["all", "train"]:
        logger.info("\n>>> STAGE 2: LOADING AND PREPROCESSING TRAINING DATA")
        s1_raw, s2_raw, s3_raw, gt_raw = load_train_data(nrows=nrows)
        gt_dict = parse_ground_truth_file(gt_raw)

        logger.info("Preprocessing train entity records...")
        s1_prep = preprocess_source_df(s1_raw)
        s2_prep = preprocess_source_df(s2_raw)
        s3_prep = preprocess_source_df(s3_raw)
        targets_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)

        logger.info("\n>>> STAGE 3: INVERTED INDEX BLOCKING & CANDIDATE GENERATION")
        candidates_dict = generate_candidates_for_dataset(
            s1_preprocessed=s1_prep,
            s2_preprocessed=s2_prep,
            s3_preprocessed=s3_prep,
        )
        evaluate_blocking_recall(candidates_dict, gt_raw)

        logger.info("\n>>> STAGE 4: FEATURE ENGINEERING, MODEL TRAINING & F0.5 THRESHOLD OPTIMIZATION")
        train_matcher_model(
            candidates_dict=candidates_dict,
            s1_preprocessed=s1_prep,
            targets_preprocessed=targets_prep,
            ground_truth_dict=gt_dict,
            model_type=args.model_type
        )
        if args.stage == "train":
            return

    # Stage: Test Prediction
    if args.stage in ["all", "predict"]:
        logger.info("\n>>> STAGE 5: TEST PREDICTION AND SUBMISSION TSV GENERATION")
        run_predict_pipeline(
            nrows=test_nrows,
            threshold=args.threshold
        )
        if args.stage == "predict":
            return

    # Stage: Official Submission Validation
    if args.stage in ["all", "validate"]:
        logger.info("\n>>> STAGE 6: OFFICIAL SUBMISSION FORMAT VALIDATION")
        is_valid = run_validator()
        if is_valid:
            logger.info("★ VALIDATION PASSED! Submission files are 100% compliant and ready.")
        else:
            logger.error("✖ VALIDATION FAILED! Please check errors above.")


if __name__ == "__main__":
    main()
