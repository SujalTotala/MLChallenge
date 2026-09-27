"""
High-performance pairwise feature engineering using RapidFuzz.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from tqdm import tqdm
from src.utils import logger, time_block

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
    "name_length_diff",
    "name_length_ratio",
    "address_exact_match",
    "address_levenshtein_ratio",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_numbers_exact",
    "address_numbers_jaccard",
    "address_numbers_overlap_count",
    "country_exact_match",
    "is_s2",
    "blocking_score",
    "blocking_rank",
]


def compute_pair_features(
    s1_dict: Dict[str, Any],
    cand_dict: Dict[str, Any],
    blocking_score: float = 0.0,
    blocking_rank: int = 0
) -> List[float]:
    """
    Compute pairwise similarity features between one S1 record and one Candidate record.
    """
    # Name features
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

    len1, len2 = len(s1_name), len(c_name)
    name_len_diff = float(abs(len1 - len2))
    name_len_ratio = (min(len1, len2) / max(1, max(len1, len2))) if (len1 or len2) else 0.0

    # Address features
    s1_addr = s1_dict.get("address_norm", "")
    c_addr = cand_dict.get("address_norm", "")

    s1_addr_tokens = set(s1_dict.get("address_tokens", []))
    c_addr_tokens = set(cand_dict.get("address_tokens", []))

    s1_nums = set(s1_dict.get("address_numbers", []))
    c_nums = set(cand_dict.get("address_numbers", []))

    addr_exact = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_tset = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

    addr_inter = len(s1_addr_tokens & c_addr_tokens)
    addr_union = len(s1_addr_tokens | c_addr_tokens)
    addr_jaccard = (addr_inter / addr_union) if addr_union > 0 else 0.0

    nums_inter = len(s1_nums & c_nums)
    nums_union = len(s1_nums | c_nums)
    nums_exact = 1.0 if (s1_nums and s1_nums == c_nums) else 0.0
    nums_jaccard = (nums_inter / nums_union) if nums_union > 0 else 0.0
    nums_overlap = float(nums_inter)

    # Country & Metadata
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
        name_len_diff,
        name_len_ratio,
        addr_exact,
        addr_lev,
        addr_tsort,
        addr_tset,
        addr_jaccard,
        nums_exact,
        nums_jaccard,
        nums_overlap,
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
    # Create fast dictionary lookups by entity_id
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
