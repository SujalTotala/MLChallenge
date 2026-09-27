"""
Rich 45-dimensional pairwise feature engineering engine for Business Entity Resolution.
Uses RapidFuzz for high-speed string metrics, TF-IDF cosine similarities, geo-conflict indicators, and interaction terms.
"""

from typing import Dict, List, Tuple, Any, Set
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from tqdm import tqdm

from src.utils import logger, time_block

FEATURE_NAMES = [
    # 1. Name Similarity Features
    "name_exact_norm",
    "name_exact_core",
    "name_exact_compact",
    "name_levenshtein_ratio",
    "name_jaro_winkler",
    "name_wratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_partial_ratio",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_token_containment",
    "name_token_count_diff",
    "name_char_len_diff",
    "name_char_len_ratio",
    
    # 2. Address Similarity Features
    "has_s1_addr",
    "has_c_addr",
    "both_have_addr",
    "address_exact_norm",
    "address_levenshtein_ratio",
    "address_jaro_winkler",
    "address_token_sort_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_token_overlap_count",
    "address_containment",
    
    # 3. Structured & Geo Indicator Features
    "house_number_exact_match",
    "house_number_jaccard",
    "house_number_conflict",
    "postal_code_match",
    "postal_code_conflict",
    "state_match",
    "state_conflict",
    "country_exact_match",

    # 4. Context & Blocking Features
    "blocking_prelim_score",
    "blocking_rank",
    "channel_support_count",
    "total_s1_candidates",
    "is_s2_target",
    "is_s3_target",

    # 5. Non-Linear Interaction Features
    "name_sim_x_address_sim",
    "name_sim_plus_address_sim",
    "name_exact_x_house_num_match",
    "name_jw_x_addr_jaccard",
    "name_exact_x_postal_match",
]


def compute_pair_features(
    s1_dict: Dict[str, Any],
    cand_dict: Dict[str, Any],
    prelim_score: float = 0.0,
    blocking_rank: int = 0,
    channel_support: int = 1,
    total_candidates: int = 1
) -> List[float]:
    """
    Compute complete feature vector for one (S1, Candidate) pair.
    """
    # --- 1. NAME FEATURES ---
    s1_norm = s1_dict.get("name_norm", "")
    c_norm = cand_dict.get("name_norm", "")

    s1_core = s1_dict.get("name_core", "")
    c_core = cand_dict.get("name_core", "")

    s1_compact = s1_dict.get("name_compact", "")
    c_compact = cand_dict.get("name_compact", "")

    s1_tokens = set(s1_dict.get("name_tokens", []))
    c_tokens = set(cand_dict.get("name_tokens", []))

    name_exact_norm = 1.0 if (s1_norm and s1_norm == c_norm) else 0.0
    name_exact_core = 1.0 if (s1_core and s1_core == c_core) else 0.0
    name_exact_compact = 1.0 if (s1_compact and s1_compact == c_compact) else 0.0

    name_lev = fuzz.ratio(s1_core, c_core) / 100.0 if s1_core and c_core else 0.0
    name_jw = distance.JaroWinkler.similarity(s1_core, c_core) if s1_core and c_core else 0.0
    name_wratio = fuzz.WRatio(s1_core, c_core) / 100.0 if s1_core and c_core else 0.0
    name_tsort = fuzz.token_sort_ratio(s1_core, c_core) / 100.0 if s1_core and c_core else 0.0
    name_tset = fuzz.token_set_ratio(s1_core, c_core) / 100.0 if s1_core and c_core else 0.0
    name_partial = fuzz.partial_ratio(s1_core, c_core) / 100.0 if s1_core and c_core else 0.0

    tok_inter = len(s1_tokens & c_tokens)
    tok_union = len(s1_tokens | c_tokens)
    name_jaccard = (tok_inter / tok_union) if tok_union > 0 else 0.0
    name_overlap = float(tok_inter)
    min_tokens = min(len(s1_tokens), len(c_tokens))
    name_containment = (tok_inter / min_tokens) if min_tokens > 0 else 0.0
    name_tok_diff = float(abs(len(s1_tokens) - len(c_tokens)))

    len1, len2 = len(s1_core), len(c_core)
    char_len_diff = float(abs(len1 - len2))
    char_len_ratio = (min(len1, len2) / max(1, max(len1, len2))) if (len1 or len2) else 0.0

    # --- 2. ADDRESS FEATURES ---
    s1_addr = s1_dict.get("address_norm", "")
    c_addr = cand_dict.get("address_norm", "")

    s1_addr_tokens = set(s1_dict.get("address_tokens", []))
    c_addr_tokens = set(cand_dict.get("address_tokens", []))

    has_s1_addr = 1.0 if s1_addr else 0.0
    has_c_addr = 1.0 if c_addr else 0.0
    both_have_addr = 1.0 if (s1_addr and c_addr) else 0.0

    addr_exact_norm = 1.0 if (s1_addr and s1_addr == c_addr) else 0.0
    addr_lev = fuzz.ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_jw = distance.JaroWinkler.similarity(s1_addr, c_addr) if s1_addr and c_addr else 0.0
    addr_tsort = fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0
    addr_tset = fuzz.token_set_ratio(s1_addr, c_addr) / 100.0 if s1_addr and c_addr else 0.0

    a_inter = len(s1_addr_tokens & c_addr_tokens)
    a_union = len(s1_addr_tokens | c_addr_tokens)
    addr_jaccard = (a_inter / a_union) if a_union > 0 else 0.0
    addr_overlap = float(a_inter)
    min_a_tokens = min(len(s1_addr_tokens), len(c_addr_tokens))
    addr_containment = (a_inter / min_a_tokens) if min_a_tokens > 0 else 0.0

    # --- 3. STRUCTURED & GEO FEATURES ---
    s1_nums = set(s1_dict.get("house_number", []))
    c_nums = set(cand_dict.get("house_number", []))

    num_exact = 1.0 if (s1_nums and s1_nums == c_nums) else 0.0
    num_union = len(s1_nums | c_nums)
    num_jaccard = (len(s1_nums & c_nums) / num_union) if num_union > 0 else 0.0
    num_conflict = 1.0 if (s1_nums and c_nums and not (s1_nums & c_nums)) else 0.0

    s1_post = s1_dict.get("postal_code", "")
    c_post = cand_dict.get("postal_code", "")
    postal_match = 1.0 if (s1_post and s1_post == c_post) else 0.0
    postal_conflict = 1.0 if (s1_post and c_post and s1_post != c_post) else 0.0

    s1_st = s1_dict.get("state", "")
    c_st = cand_dict.get("state", "")
    state_match = 1.0 if (s1_st and s1_st == c_st) else 0.0
    state_conflict = 1.0 if (s1_st and c_st and s1_st != c_st) else 0.0

    s1_country = s1_dict.get("country", "")
    c_country = cand_dict.get("country", "")
    country_match = 1.0 if (s1_country and s1_country == c_country) else 0.0

    # --- 4. CONTEXT & METADATA ---
    cand_id = cand_dict.get("entity_id", "")
    is_s2 = 1.0 if str(cand_id).startswith("S2-") else 0.0
    is_s3 = 1.0 if str(cand_id).startswith("S3-") else 0.0

    # --- 5. INTERACTION TERMS ---
    name_x_addr = name_lev * addr_lev if both_have_addr else name_lev * 0.5
    name_plus_addr = name_lev + addr_lev if both_have_addr else name_lev
    name_exact_x_house = name_exact_core * num_exact
    name_jw_x_addr_j = name_jw * addr_jaccard if both_have_addr else 0.0
    name_exact_x_postal = name_exact_core * postal_match

    return [
        name_exact_norm,
        name_exact_core,
        name_exact_compact,
        name_lev,
        name_jw,
        name_wratio,
        name_tsort,
        name_tset,
        name_partial,
        name_jaccard,
        name_overlap,
        name_containment,
        name_tok_diff,
        char_len_diff,
        char_len_ratio,
        has_s1_addr,
        has_c_addr,
        both_have_addr,
        addr_exact_norm,
        addr_lev,
        addr_jw,
        addr_tsort,
        addr_tset,
        addr_jaccard,
        addr_overlap,
        addr_containment,
        num_exact,
        num_jaccard,
        num_conflict,
        postal_match,
        postal_conflict,
        state_match,
        state_conflict,
        country_match,
        float(prelim_score),
        float(blocking_rank),
        float(channel_support),
        float(total_candidates),
        is_s2,
        is_s3,
        name_x_addr,
        name_plus_addr,
        name_exact_x_house,
        name_jw_x_addr_j,
        name_exact_x_postal,
    ]


@time_block("Pairwise Feature Matrix Extraction")
def extract_features_for_candidates(
    candidates_dict: Dict[str, List[Tuple[str, float, int]]],
    s1_preprocessed: pd.DataFrame,
    targets_preprocessed: pd.DataFrame,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """
    Extract matrix X of pairwise features and aligned list of (s1_id, candidate_id) pairs.
    """
    logger.info("Building fast index for entity feature retrieval...")
    s1_map = {row.entity_id: row._asdict() for row in s1_preprocessed.itertuples(index=False)}
    targets_map = {row.entity_id: row._asdict() for row in targets_preprocessed.itertuples(index=False)}

    feature_rows = []
    pair_ids = []

    total_pairs = sum(len(cands) for cands in candidates_dict.values())
    logger.info(f"Extracting features for {total_pairs:,} candidate pairs...")

    with tqdm(total=total_pairs, desc="Feature Extraction") as pbar:
        for s1_id, cands in candidates_dict.items():
            if s1_id not in s1_map:
                pbar.update(len(cands))
                continue
            s1_info = s1_map[s1_id]
            total_cand_count = len(cands)

            for rank, (cand_id, score, channel_support) in enumerate(cands):
                if cand_id not in targets_map:
                    pbar.update(1)
                    continue
                cand_info = targets_map[cand_id]

                feat_vec = compute_pair_features(
                    s1_dict=s1_info,
                    cand_dict=cand_info,
                    prelim_score=score,
                    blocking_rank=rank,
                    channel_support=channel_support,
                    total_candidates=total_cand_count
                )
                feature_rows.append(feat_vec)
                pair_ids.append((s1_id, cand_id))
                pbar.update(1)

    X = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    return X, pair_ids
