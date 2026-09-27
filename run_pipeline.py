"""
Master Execution Pipeline for Amazon ML Challenge 2026: Business Entity Resolution.
Supports full dataset execution (--mode full) and rapid development mode (--mode sample).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    TRAIN_SOURCE1_PATH,
    OUTPUT_MATCHING_PATH,
    OUTPUT_CANDIDATES_PATH,
    TEST_DIR,
    METADATA_PATH,
    REPORTS_DIR,
)
from src.data_loader import load_train_data, preprocess_source_df
from src.candidate_generation import generate_candidates_for_dataset, evaluate_blocking_recall
from src.train_model import train_matcher_model
from src.predict import run_predict_pipeline
from src.eda import run_eda
from src.utils import parse_ground_truth_file, logger, time_block

BLOCKING_REPORT_CSV = REPORTS_DIR / "blocking_report.csv"
VALIDATION_RESULTS_CSV = REPORTS_DIR / "validation_results.csv"


def run_official_validator() -> bool:
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
    logger.info(f"Running official validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


@time_block("Complete Entity Resolution Pipeline")
def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026: Business Entity Resolution Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "sample"],
        default="full",
        help="Pipeline mode: 'full' uses ALL dataset records, 'sample' uses fast sample subset (default: full)"
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
        help="Number of records to sample when in sample mode (default: 25000)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Max candidates per S1 entity during blocking (default: 25)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision threshold for prediction"
    )
    args = parser.parse_args()

    nrows = None if args.mode == "full" else args.sample_size

    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    logger.info(f"Mode: {args.mode.upper()} | Stage: {args.stage} | Sample Size: {nrows or 'FULL'} | Top-K: {args.top_k}")
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

        logger.info("\n>>> STAGE 3: MULTI-CHANNEL BLOCKING & CANDIDATE GENERATION")
        candidates_dict = generate_candidates_for_dataset(
            s1_preprocessed=s1_prep,
            s2_preprocessed=s2_prep,
            s3_preprocessed=s3_prep,
            top_k=args.top_k
        )
        blocking_stats = evaluate_blocking_recall(candidates_dict, gt_raw, s1_preprocessed=s1_prep)

        logger.info("\n>>> STAGE 4: FEATURE ENGINEERING, MODEL TRAINING & SINGLETON OPTIMIZATION")
        train_matcher_model(
            candidates_dict=candidates_dict,
            s1_preprocessed=s1_prep,
            targets_preprocessed=targets_prep,
            ground_truth_dict=gt_dict,
            model_type="lightgbm"
        )
        if args.stage == "train":
            return

    # Stage: Test Prediction
    if args.stage in ["all", "predict"]:
        logger.info("\n>>> STAGE 5: TEST PREDICTION AND SUBMISSION TSV GENERATION")
        run_predict_pipeline(
            nrows=nrows,
            threshold=args.threshold
        )
        if args.stage == "predict":
            return

    # Stage: Official Submission Validation
    if args.stage in ["all", "validate"]:
        logger.info("\n>>> STAGE 6: OFFICIAL SUBMISSION FORMAT VALIDATION")
        is_valid = run_official_validator()

        # Load metrics dynamically from generated reports
        macro_f05 = "N/A"
        best_thresh = "N/A"
        cand_recall = "N/A"
        entity_complete_rec = "N/A"
        avg_cands = "N/A"
        p95_cands = "N/A"
        reduction_ratio = "N/A"
        singleton_acc = "N/A"

        if BLOCKING_REPORT_CSV.exists():
            b_df = pd.read_csv(BLOCKING_REPORT_CSV)
            if not b_df.empty:
                r = b_df.iloc[0]
                cand_recall = f"{r.get('pair_candidate_recall', 0.0):.4f}"
                entity_complete_rec = f"{r.get('entity_complete_recall', 0.0):.4f}"
                avg_cands = f"{r.get('mean_candidates_per_s1', 0.0):.2f}"
                p95_cands = f"{r.get('p95_candidates_per_s1', 0.0):.1f}"
                reduction_ratio = f"{r.get('candidate_reduction_ratio', 0.0):.8f}"

        if VALIDATION_RESULTS_CSV.exists():
            v_df = pd.read_csv(VALIDATION_RESULTS_CSV)
            if not v_df.empty:
                vr = v_df.iloc[0]
                macro_f05 = f"{vr.get('macro_f05', 0.0):.4f}"
                best_thresh = f"{vr.get('best_threshold', 0.0):.2f}"
                singleton_acc = f"{vr.get('singleton_accuracy', 0.0):.4f}"

        print("\n" + "=" * 60)
        print("FINAL PIPELINE SUMMARY")
        print("=" * 60)
        print(f"Validation Macro F0.5:         {macro_f05}")
        print(f"Best threshold:               {best_thresh}")
        print(f"Candidate recall:             {cand_recall}")
        print(f"Entity-complete recall:        {entity_complete_rec}")
        print(f"Average candidates/S1:        {avg_cands}")
        print(f"P95 candidates/S1:            {p95_cands}")
        print(f"Candidate reduction ratio:    {reduction_ratio}")
        print(f"Singleton accuracy:           {singleton_acc}")
        print("\nOfficial validator:")
        print("PASS" if is_valid else "FAIL")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
