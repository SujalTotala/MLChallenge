"""
Data loading and ingestion utilities for TSV datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
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
    Load a tab-separated values file safely.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        nrows=nrows,
        keep_default_na=False,
        encoding="utf-8",
    )
    # Strip any column whitespace
    df.columns = [c.strip() for c in df.columns]
    return df


def load_filtered_target_tsv(path: Path, target_ids: Set[str], max_distractors: int = 50000) -> pd.DataFrame:
    """
    Stream a target TSV to collect all matching target_ids plus up to max_distractors random records.
    """
    rows = []
    distractors_added = 0
    with open(path, "r", encoding="utf-8") as f:
        header_line = next(f, "").strip()
        headers = [c.strip() for c in header_line.split("\t")]
        for line in f:
            if not line.strip():
                continue
            parts = [p.strip() for p in line.rstrip("\n").split("\t")]
            if not parts:
                continue
            eid = parts[0]
            if eid in target_ids:
                rows.append(parts[:len(headers)])
            elif distractors_added < max_distractors:
                rows.append(parts[:len(headers)])
                distractors_added += 1

    return pd.DataFrame(rows, columns=headers)


@time_block("Preprocessing DataFrame")
def preprocess_source_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text normalization and extract feature keys across an entity dataframe.
    """
    # Detect standard column names or fallback
    id_col = next((c for c in ["entity_id", "source1_entity_id", "id"] if c in df.columns), df.columns[0])
    name_col = next((c for c in ["business_name", "name", "company_name"] if c in df.columns), None)
    addr_col = next((c for c in ["business_address", "address", "addr"] if c in df.columns), None)
    country_col = next((c for c in ["country", "cntry", "nation"] if c in df.columns), None)

    names = df[name_col].fillna("").astype(str).tolist() if name_col else [""] * len(df)
    addrs = df[addr_col].fillna("").astype(str).tolist() if addr_col else [""] * len(df)
    countries = df[country_col].fillna("").astype(str).tolist() if country_col else [""] * len(df)
    ids = df[id_col].astype(str).str.strip().tolist()

    records = []
    for eid, name, addr, cntry in zip(ids, names, addrs, countries):
        n_info = normalize_name(name)
        a_info = normalize_address(addr)
        c_norm = normalize_country(cntry)
        records.append({
            "entity_id": eid,
            "raw_name": name,
            "raw_address": addr,
            "country": c_norm,
            "name_norm": n_info["name_norm"],
            "name_no_suffix": n_info["name_no_suffix"],
            "name_alnum": n_info["name_alnum"],
            "name_tokens": n_info["name_tokens"],
            "address_norm": a_info["address_norm"],
            "address_alnum": a_info["address_alnum"],
            "address_tokens": a_info["address_tokens"],
            "address_numbers": a_info["address_numbers"],
        })

    return pd.DataFrame(records)


def load_train_data(nrows: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all training datasets and ground truth.
    When nrows is specified, aligns ground truth by entity_id and loads target records accurately.
    """
    logger.info(f"Loading training data (nrows={nrows})...")
    s1_df = load_raw_tsv(TRAIN_SOURCE1_PATH, nrows=nrows)
    s1_id_set = set(s1_df["entity_id"])

    if nrows is not None and nrows < 2000000:
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
        logger.info(f"Loaded {len(s1_df):,} Train S1 entities with {len(target_ids):,} true target IDs")
        s2_df = load_filtered_target_tsv(TRAIN_SOURCE2_PATH, target_ids, max_distractors=nrows * 3)
        s3_df = load_filtered_target_tsv(TRAIN_SOURCE3_PATH, target_ids, max_distractors=nrows * 3)
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
    logger.info(f"Loading test data (nrows={nrows})...")
    s1_df = load_raw_tsv(TEST_SOURCE1_PATH, nrows=nrows)
    s2_df = load_raw_tsv(TEST_SOURCE2_PATH, nrows=nrows * 4 if nrows else None)
    s3_df = load_raw_tsv(TEST_SOURCE3_PATH, nrows=nrows * 4 if nrows else None)

    return s1_df, s2_df, s3_df
