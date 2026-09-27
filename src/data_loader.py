"""
Data loading, ingestion, and preprocessing pipeline for TSV datasets.
Includes automatic quality inspection, schema validation, and missing value checks.
Ensures sample mode loads all ground truth target entities for aligned validation evaluation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
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
)
from src.normalize import normalize_name, normalize_address, normalize_country
from src.utils import logger, time_block


def load_raw_tsv(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load a tab-separated values file safely into a pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required TSV file not found at: {path}")

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        keep_default_na=False,
        encoding="utf-8",
    )
    df.columns = [c.strip() for c in df.columns]
    return df


def inspect_dataset(df: pd.DataFrame, name: str = "Dataset") -> Dict[str, Any]:
    """
    Automatically inspect dataset metrics:
    - row counts
    - missing values
    - duplicate IDs
    - country distribution
    - name length distribution
    - address length distribution
    - percentage of empty names
    - percentage of empty addresses
    """
    id_col = next((c for c in ["entity_id", "source1_entity_id", "id"] if c in df.columns), df.columns[0])
    name_col = next((c for c in ["business_name", "name", "company_name"] if c in df.columns), None)
    addr_col = next((c for c in ["business_address", "address", "addr"] if c in df.columns), None)
    country_col = next((c for c in ["country", "cntry", "nation"] if c in df.columns), None)

    total_rows = len(df)
    unique_ids = df[id_col].nunique()
    duplicate_ids = total_rows - unique_ids

    empty_names = (df[name_col].astype(str).str.strip() == "").sum() if name_col else total_rows
    empty_addrs = (df[addr_col].astype(str).str.strip() == "").sum() if addr_col else total_rows

    pct_empty_name = (empty_names / total_rows * 100) if total_rows > 0 else 0.0
    pct_empty_addr = (empty_addrs / total_rows * 100) if total_rows > 0 else 0.0

    name_word_lens = df[name_col].astype(str).apply(lambda x: len(x.split())).tolist() if name_col else []
    addr_word_lens = df[addr_col].astype(str).apply(lambda x: len(x.split())).tolist() if addr_col else []

    country_counts = df[country_col].value_counts().to_dict() if country_col else {}

    metrics = {
        "dataset_name": name,
        "total_rows": total_rows,
        "unique_ids": unique_ids,
        "duplicate_ids": duplicate_ids,
        "pct_empty_name": pct_empty_name,
        "pct_empty_addr": pct_empty_addr,
        "country_distribution": country_counts,
        "mean_name_len": float(np.mean(name_word_lens)) if name_word_lens else 0.0,
        "mean_addr_len": float(np.mean(addr_word_lens)) if addr_word_lens else 0.0,
    }

    logger.info(f"--- DATASET INSPECTION REPORT: {name} ---")
    logger.info(f"  Row count: {total_rows:,} | Unique IDs: {unique_ids:,} | Duplicate IDs: {duplicate_ids}")
    logger.info(f"  Empty Names: {empty_names:,} ({pct_empty_name:.2f}%) | Empty Addresses: {empty_addrs:,} ({pct_empty_addr:.2f}%)")
    if country_counts:
        logger.info(f"  Country breakdown: {country_counts}")
    if name_word_lens:
        logger.info(f"  Name word count mean: {metrics['mean_name_len']:.1f} | Address word count mean: {metrics['mean_addr_len']:.1f}")

    return metrics


@time_block("Entity Records Preprocessing")
def preprocess_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply multi-representation text normalization across entity dataframe.
    """
    id_col = next((c for c in ["entity_id", "source1_entity_id", "id"] if c in df.columns), df.columns[0])
    name_col = next((c for c in ["business_name", "name", "company_name"] if c in df.columns), None)
    addr_col = next((c for c in ["business_address", "address", "addr"] if c in df.columns), None)
    country_col = next((c for c in ["country", "cntry", "nation"] if c in df.columns), None)

    ids = df[id_col].astype(str).str.strip().tolist()
    names = df[name_col].tolist() if name_col else [""] * len(df)
    addrs = df[addr_col].tolist() if addr_col else [""] * len(df)
    countries = df[country_col].tolist() if country_col else [""] * len(df)

    records = []
    for eid, name, addr, cntry in zip(ids, names, addrs, countries):
        c_norm = normalize_country(cntry)
        n_info = normalize_name(name)
        a_info = normalize_address(addr, country_norm=c_norm)

        records.append({
            "entity_id": eid,
            "country": c_norm,
            "name_raw": n_info["name_raw"],
            "name_norm": n_info["name_norm"],
            "name_core": n_info["name_core"],
            "name_compact": n_info["name_compact"],
            "name_tokens": n_info["name_tokens"],
            "name_sorted": n_info["name_sorted"],
            "name_first_tokens": n_info["name_first_tokens"],
            "address_norm": a_info["address_norm"],
            "address_tokens": a_info["address_tokens"],
            "house_number": a_info["house_number"],
            "postal_code": a_info["postal_code"],
            "state": a_info["state"],
            "street_tokens": a_info["street_tokens"],
            "address_compact": a_info["address_compact"],
        })

    return pd.DataFrame(records)


def load_train_data(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all training datasets and ground truth.
    Supports both sample mode (aligned target pool for accurate recall evaluation) and full mode.
    """
    logger.info(f"Loading training data (nrows={nrows or 'FULL'})...")
    s1_df = load_raw_tsv(TRAIN_SOURCE1_PATH, nrows=nrows)
    inspect_dataset(s1_df, "Train Source 1")

    if nrows is not None:
        s1_id_set = set(s1_df["entity_id"])
        target_ids = set()
        aligned_gt_rows = []
        with open(TRAIN_GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                if not line.strip():
                    continue
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0] in s1_id_set:
                    aligned_gt_rows.append(parts[:2])
                    if len(parts) > 1 and parts[1].strip() and parts[1].lower() not in {"nan", "none"}:
                        for mid in parts[1].split(","):
                            if mid.strip():
                                target_ids.add(mid.strip())

        gt_df = pd.DataFrame(aligned_gt_rows, columns=["source1_entity_id", "matched_entity_ids"])
        
        # Load S2 and S3 containing all true targets + background sample
        s2_full = load_raw_tsv(TRAIN_SOURCE2_PATH)
        s3_full = load_raw_tsv(TRAIN_SOURCE3_PATH)

        s2_true = s2_full[s2_full["entity_id"].isin(target_ids)]
        s3_true = s3_full[s3_full["entity_id"].isin(target_ids)]

        s2_bg = s2_full[~s2_full["entity_id"].isin(target_ids)].iloc[:nrows * 2]
        s3_bg = s3_full[~s3_full["entity_id"].isin(target_ids)].iloc[:nrows * 2]

        s2_df = pd.concat([s2_true, s2_bg], ignore_index=True).drop_duplicates(subset=["entity_id"])
        s3_df = pd.concat([s3_true, s3_bg], ignore_index=True).drop_duplicates(subset=["entity_id"])
    else:
        gt_df = load_raw_tsv(TRAIN_GROUND_TRUTH_PATH)
        s2_df = load_raw_tsv(TRAIN_SOURCE2_PATH)
        s3_df = load_raw_tsv(TRAIN_SOURCE3_PATH)

    logger.info(f"Loaded datasets: S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}, GT={len(gt_df):,}")
    return s1_df, s2_df, s3_df, gt_df


def load_test_data(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all test datasets.
    """
    logger.info(f"Loading test data (nrows={nrows or 'FULL'})...")
    s1_df = load_raw_tsv(TEST_SOURCE1_PATH, nrows=nrows)
    inspect_dataset(s1_df, "Test Source 1")

    s2_df = load_raw_tsv(TEST_SOURCE2_PATH, nrows=nrows * 4 if nrows else None)
    s3_df = load_raw_tsv(TEST_SOURCE3_PATH, nrows=nrows * 4 if nrows else None)

    logger.info(f"Loaded test datasets: S1={len(s1_df):,}, S2={len(s2_df):,}, S3={len(s3_df):,}")
    return s1_df, s2_df, s3_df
