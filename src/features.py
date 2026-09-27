"""
High-performance pairwise feature engineering using RapidFuzz and domain-specific geo/entity indicators.
Optimized for high precision and Macro F0.5.
"""

import re
from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from tqdm import tqdm
from src.utils import logger, time_block

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "dc", "pr"
}

RE_PINCODE_IN = re.compile(r"\b[1-9][0-9]{5}\b")
RE_ZIPCODE_US = re.compile(r"\b\d{5}\b")

FEATURE_NAMES = [
    "name_exact_match",
    "name_no_suffix_exact",
    "name_alnum_exact",
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_token_containment",
    "name_token_diff_count",
    "name_length_diff",
    "name_length_ratio",
    "name_num_tokens",
    "has_s1_addr",
    "has_c_addr",
    "both_have_addr",
    "one_addr_missing",
    "address_exact_match",
    "address_levenshtein_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_containment",
    "address_numbers_exact",
    "address_numbers_jaccard",
    "address_numbers_overlap_count",
    "number_conflict",
    "pincode_match",
    "pincode_conflict",
    "zipcode_match",
    "zipcode_conflict",
    "state_match",
    "state_conflict",
    "name_x_addr_sim",
    "name_jw_x_addr_jaccard",
    "country_exact_match",
    "is_s2",
    "blocking_score",
    "blocking_rank",
]


def extract_geo_markers(addr_norm: str, tokens: List[str], country: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Extract Indian 6-digit pincodes, US 5-digit zip codes, and US state 2-letter abbreviations."""
    pincodes = []
    zipcodes = []
    states = []
    
    if country == "india" and addr_norm:
        pincodes = RE_PINCODE_IN.findall(addr_norm)
    elif country == "us" and addr_norm:
        zipcodes = RE_ZIPCODE_US.findall(addr_norm)
        for t in tokens:
            if t in US_STATES:
                states.append(t)
    return set(pincodes), set(zipcodes), set(states)


def compute_pair_features(
    s1_dict: Dict[str, Any],
    cand_dict: Dict[str, Any],
    blocking_score: float = 0.0,
    blocking_rank: int = 0
) -> List[float]:
    """
    Compute 40-dimensional pairwise similarity features between one S1 record and one Candidate record.
    Includes precision-critical geo-conflict and address discrepancy indicators.
    """
    # 1. Name features
    s1_name = s1_dict.get("name_norm", "")
    c_name = cand_dict.get("name_norm", "")
    
    s1_name_ns = s1_dict.get("name_no_suffix", "")
    c_name_ns = cand_dict.get("name_no_suffix", "")

    s1_name_alnum = s1_dict.get("name_alnum", "")
    c_name_alnum = cand_dict.get("name_alnum", "")

    s1_tokens = set(s1_dict.get("name_tokens", []))
    c_tokens = set(cand_dict.get("name_tokens", []))

    name_exact = 1.0 if (s1_name and s1_name == c_name) else 0.0
    name_ns_exact = 1.0 if (s1_name_ns and s1_name_ns == c_name_ns) else 0.0
    name_alnum_exact = 1.0 if (s1_name_alnum and s1_name_alnum == c_name_alnum) else 0.0

    name_lev = fuzz.ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_name, c_name) if s1_name and c_name else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0
    name_tset = fuzz.token_set_ratio(s1_name, c_name) / 100.0 if s1_name and c_name else 0.0

    tok_inter = len(s1_tokens & c_tokens)
    tok_union = len(s1_tokens | c_tokens)
    name_jaccard = (tok_inter / tok_union) if tok_union > 0 else 0.0
    name_overlap = float(tok_inter)
    min_len = min(len(s1_tokens), len(c_tokens))
    name_containment = (tok_inter / min_len) if min_len > 0 else 0.0
    tok_diff_count = float(len(s1_tokens ^ c_tokens))

    len1, len2 = len(s1_name), len(c_name)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = (min(len1, len2) / max(1, max(len1, len2))) if (len1 or len2) else 0.0
    name_num_tokens = float(len(s1_tokens))

    # 2. Address features
    s1_addr = s1_dict.get("address_norm", "")
    c_addr = cand_dict.get("address_norm", "")

    s1_addr_tokens = set(s1_dict.get("address_tokens", []))
    c_addr_tokens = set(cand_dict.get("address_tokens", []))

    s1_nums = set(s1_dict.get("address_numbers", []))
    c_nums = set(cand_dict.get("address_numbers", []))

    has_s1_addr = 1.0 if s1_addr else 0.0
    has_c_addr = 1.0 if c_addr else 0.0
    both_have_addr = 1.0 if (s1_addr and c_addr) else 0.0
    one_addr_missing = 1.0 if ((s1_addr and not c_addr) or (not s1_addr and c_addr)) else 0.0

    addr_exact = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_tset = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

    addr_inter = len(s1_addr_tokens & c_addr_tokens)
    addr_union = len(s1_addr_tokens | c_addr_tokens)
    addr_jaccard = (addr_inter / addr_union) if addr_union > 0 else 0.0
    addr_containment = (addr_inter / min(len(s1_addr_tokens), len(c_addr_tokens))) if (s1_addr_tokens and c_addr_tokens) else 0.0

    nums_inter = len(s1_nums & c_nums)
    nums_union = len(s1_nums | c_nums)
    nums_exact = 1.0 if (s1_nums and s1_nums == c_nums) else 0.0
    nums_jaccard = (nums_inter / nums_union) if nums_union > 0 else 0.0
    nums_overlap = float(nums_inter)

    # Number conflict: both have numbers, but set intersection is empty (different street numbers)
    number_conflict = 1.0 if (len(s1_nums) > 0 and len(c_nums) > 0 and nums_inter == 0) else 0.0

    # 3. Geo Markers & Conflict Detection
    country = s1_dict.get("country", "")
    s1_pins, s1_zips, s1_states = extract_geo_markers(s1_addr, s1_dict.get("address_tokens", []), country)
    c_pins, c_zips, c_states = extract_geo_markers(c_addr, cand_dict.get("address_tokens", []), country)

    pincode_match = 1.0 if (s1_pins and c_pins and (s1_pins & c_pins)) else 0.0
    pincode_conflict = 1.0 if (s1_pins and c_pins and not (s1_pins & c_pins)) else 0.0

    zipcode_match = 1.0 if (s1_zips and c_zips and (s1_zips & c_zips)) else 0.0
    zipcode_conflict = 1.0 if (s1_zips and c_zips and not (s1_zips & c_zips)) else 0.0

    state_match = 1.0 if (s1_states and c_states and (s1_states & c_states)) else 0.0
    state_conflict = 1.0 if (s1_states and c_states and not (s1_states & c_states)) else 0.0

    # 4. Non-linear Interaction Features
    name_x_addr = name_lev * addr_lev if both_have_addr else name_lev * 0.5
    name_jw_x_addr_jaccard = name_jw * addr_jaccard if both_have_addr else 0.0

    # 5. Metadata
    s1_cntry = s1_dict.get("country", "")
    c_cntry = cand_dict.get("country", "")
    country_match = 1.0 if (s1_cntry and s1_cntry == c_cntry) else 0.0

    cand_id = cand_dict.get("entity_id", "")
    is_s2 = 1.0 if str(cand_id).startswith("S2-") else 0.0

    return [
        name_exact,
        name_ns_exact,
        name_alnum_exact,
        name_lev,
        name_jw,
        name_tsort,
        name_tset,
        name_jaccard,
        name_overlap,
        name_containment,
        tok_diff_count,
        name_len_diff,
        name_len_ratio,
        name_num_tokens,
        has_s1_addr,
        has_c_addr,
        both_have_addr,
        one_addr_missing,
        addr_exact,
        addr_lev,
        addr_tsort,
        addr_tset,
        addr_jaccard,
        addr_containment,
        nums_exact,
        nums_jaccard,
        nums_overlap,
        number_conflict,
        pincode_match,
        pincode_conflict,
        zipcode_match,
        zipcode_conflict,
        state_match,
        state_conflict,
        name_x_addr,
        name_jw_x_addr_jaccard,
        country_match,
        is_s2,
        float(blocking_score),
        float(blocking_rank),
    ]


@time_block("Pairwise Feature Matrix Generation")
def extract_features_for_candidates(
    candidates_dict: Dict[str, List[Tuple[str, float]]],
    s1_preprocessed: pd.DataFrame,
    targets_preprocessed: pd.DataFrame,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Extract pairwise feature matrix X and corresponding pair IDs (s1_id, candidate_id).
    """
    logger.info("Indexing preprocessed records for fast pairwise feature extraction...")
    s1_map = {row.entity_id: row._asdict() for row in s1_preprocessed.itertuples(index=False)}
    targets_map = {row.entity_id: row._asdict() for row in targets_preprocessed.itertuples(index=False)}

    feature_rows = []
    pair_ids = []

    total_pairs = sum(len(cands) for cands in candidates_dict.values())
    logger.info(f"Computing features for {total_pairs:,} candidate pairs...")

    with tqdm(total=total_pairs, desc="Feature Extraction") as pbar:
        for s1_id, cands in candidates_dict.items():
            if s1_id not in s1_map:
                continue
            s1_info = s1_map[s1_id]

            for rank, (cand_id, score) in enumerate(cands):
                if cand_id not in targets_map:
                    pbar.update(1)
                    continue
                cand_info = targets_map[cand_id]

                feat_vec = compute_pair_features(
                    s1_dict=s1_info,
                    cand_dict=cand_info,
                    blocking_score=score,
                    blocking_rank=rank
                )
                feature_rows.append(feat_vec)
                pair_ids.append((s1_id, cand_id))
                pbar.update(1)

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X, pair_ids
