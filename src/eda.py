"""
Exploratory Data Analysis (EDA) module for Business Entity Resolution.
Generates comprehensive analysis of dataset characteristics and writes output/eda_report.txt.
"""

from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

from src.config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    OUTPUT_EDA_REPORT_PATH,
)
from src.data_loader import load_raw_tsv
from src.utils import parse_id_list, logger, time_block


def fast_count_lines(path: Path) -> int:
    """Fast binary buffer line count."""
    lines = 0
    buf_size = 1024 * 1024
    with open(path, 'rb') as f:
        read_chunk = f.raw.read if hasattr(f, 'raw') else f.read
        buf = read_chunk(buf_size)
        while buf:
            lines += buf.count(b'\n')
            buf = read_chunk(buf_size)
    return max(0, lines - 1)


@time_block("EDA Report Generation")
def run_eda(sample_size: int = 25000) -> str:
    """
    Run comprehensive EDA on train and test datasets.
    """
    report_lines = []
    
    def log_and_record(text: str = ""):
        print(text)
        report_lines.append(text)

    log_and_record("=" * 60)
    log_and_record("AMAZON ML CHALLENGE 2026: EXPLORATORY DATA ANALYSIS")
    log_and_record("=" * 60)

    # 1. Dataset Overview & File Checks
    files_to_check = [
        ("Train Source 1", TRAIN_SOURCE1_PATH),
        ("Train Source 2", TRAIN_SOURCE2_PATH),
        ("Train Source 3", TRAIN_SOURCE3_PATH),
        ("Train Ground Truth", TRAIN_GROUND_TRUTH_PATH),
        ("Test Source 1", TEST_SOURCE1_PATH),
        ("Test Source 2", TEST_SOURCE2_PATH),
        ("Test Source 3", TEST_SOURCE3_PATH),
    ]

    log_and_record("\n[1] DATASET FILES AND SCHEMA INSPECTION:")
    for label, path in files_to_check:
        if not path.exists():
            log_and_record(f"  - {label} ({path.name}): NOT FOUND")
            continue
        
        total_lines = fast_count_lines(path)
        sample_df = load_raw_tsv(path, nrows=5)
        cols = sample_df.columns.tolist()
        log_and_record(f"  - {label} ({path.name}): ~{total_lines:,} rows | Columns: {cols}")

    # 2. Detailed Training Distribution
    log_and_record("\n[2] TRAINING DATA ANALYSIS (Sampled):")
    s1_df = load_raw_tsv(TRAIN_SOURCE1_PATH, nrows=sample_size)
    s2_df = load_raw_tsv(TRAIN_SOURCE2_PATH, nrows=sample_size)
    s3_df = load_raw_tsv(TRAIN_SOURCE3_PATH, nrows=sample_size)
    gt_df = load_raw_tsv(TRAIN_GROUND_TRUTH_PATH, nrows=sample_size)

    log_and_record(f"  Sample size analyzed: {sample_size:,} rows per source")

    for name, df in [("Source 1", s1_df), ("Source 2", s2_df), ("Source 3", s3_df)]:
        empty_str = {c: int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns}
        log_and_record(f"  - {name} Empty string counts: {empty_str}")

    # Country distribution
    log_and_record("\n[3] COUNTRY DISTRIBUTION:")
    log_and_record("  Train Source 1 Country breakdown:")
    for country, count in s1_df["country"].value_counts().items():
        log_and_record(f"    - {country}: {count:,} ({count/len(s1_df):.2%})")

    # Check test countries
    if TEST_SOURCE1_PATH.exists():
        test_s1 = load_raw_tsv(TEST_SOURCE1_PATH, nrows=sample_size)
        log_and_record("  Test Source 1 Country breakdown (Open Set Check):")
        for country, count in test_s1["country"].value_counts().items():
            log_and_record(f"    - {country}: {count:,} ({count/len(test_s1):.2%})")

    # 4. Ground Truth Match Statistics
    log_and_record("\n[4] GROUND TRUTH MATCH STATISTICS:")
    match_counts = []
    s2_matches = 0
    s3_matches = 0
    singletons = 0

    for _, row in gt_df.iterrows():
        matches = parse_id_list(row.get("matched_entity_ids"))
        match_counts.append(len(matches))
        if len(matches) == 0:
            singletons += 1
        for m in matches:
            if m.startswith("S2-"):
                s2_matches += 1
            elif m.startswith("S3-"):
                s3_matches += 1

    total_gt = len(gt_df)
    total_matches = sum(match_counts)
    log_and_record(f"  - Total Ground Truth S1 Records Analyzed: {total_gt:,}")
    log_and_record(f"  - True Singletons (0 matches): {singletons:,} ({singletons/total_gt:.2%})")
    log_and_record(f"  - Entities with >= 1 matches: {total_gt - singletons:,} ({(total_gt - singletons)/total_gt:.2%})")
    log_and_record(f"  - Total Match Links: {total_matches:,}")
    log_and_record(f"  - S2 Target Matches: {s2_matches:,} ({s2_matches/max(1, total_matches):.2%})")
    log_and_record(f"  - S3 Target Matches: {s3_matches:,} ({s3_matches/max(1, total_matches):.2%})")
    log_and_record(f"  - Average matches per S1: {np.mean(match_counts):.2f}")

    # 5. Text Characteristics
    log_and_record("\n[5] TEXT TOKEN CHARACTERISTICS:")
    name_lengths = [len(str(n).split()) for n in s1_df["business_name"]]
    addr_lengths = [len(str(a).split()) for a in s1_df["business_address"]]
    log_and_record(f"  - S1 Name words mean: {np.mean(name_lengths):.2f}")
    log_and_record(f"  - S1 Address words mean: {np.mean(addr_lengths):.2f}")

    report_text = "\n".join(report_lines)
    OUTPUT_EDA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_EDA_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"EDA report saved to {OUTPUT_EDA_REPORT_PATH}")

    return report_text


if __name__ == "__main__":
    run_eda()