"""
Multi-Channel Scalable Blocking Engine for Business Entity Resolution.
Implements Channels A through K with Inverted Indexes and Vectorized Batch TF-IDF Nearest-Neighbor Retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BLOCKING_CONFIG, TOP_K_CANDIDATES
from src.utils import logger, time_block


def compute_quick_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Fast Jaccard similarity between two token lists."""
    if not tokens1 or not tokens2:
        return 0.0
    s1 = set(tokens1)
    s2 = set(tokens2)
    intersection = len(s1 & s2)
    if intersection == 0:
        return 0.0
    return intersection / len(s1 | s2)


class CountryBlockingIndex:
    """
    Multi-channel inverted index and TF-IDF vector index for a single country partition.
    """
    def __init__(self, country: str, target_df: pd.DataFrame):
        self.country = country
        self.total_records = len(target_df)
        self.target_df = target_df.reset_index(drop=True)

        self.entity_ids = self.target_df["entity_id"].tolist()
        self.name_cores = self.target_df["name_core"].tolist()
        self.name_compacts = self.target_df["name_compact"].tolist()
        self.name_tokens = self.target_df["name_tokens"].tolist()
        self.name_first_tokens = self.target_df["name_first_tokens"].tolist()
        self.address_norms = self.target_df["address_norm"].tolist()
        self.address_tokens = self.target_df["address_tokens"].tolist()
        self.house_numbers = self.target_df["house_number"].tolist()
        self.postal_codes = self.target_df["postal_code"].tolist()
        self.states = self.target_df["state"].tolist()

        # Inverted index channels
        self.exact_core_index = defaultdict(list)     # Channel A
        self.compact_name_index = defaultdict(list)   # Channel B
        self.first_tokens_index = defaultdict(list)   # Channel C
        self.rare_name_token_index = defaultdict(list)# Channel D
        self.name_house_num_index = defaultdict(list) # Channel E
        self.house_addr_token_index = defaultdict(list)# Channel F
        self.postal_name_index = defaultdict(list)    # Channel G

        self.name_token_freq = defaultdict(int)
        self.addr_token_freq = defaultdict(int)

        self.char_tfidf = None
        self.char_tfidf_matrix = None
        self.addr_tfidf = None
        self.addr_tfidf_matrix = None

        self._build_indexes()
        self._build_tfidf_indexes()

    def _build_indexes(self):
        """Build channels A through G inverted indexes."""
        for tokens in self.name_tokens:
            for t in set(tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"]:
                    self.name_token_freq[t] += 1

        for tokens in self.address_tokens:
            for t in set(tokens):
                if len(t) >= 4:
                    self.addr_token_freq[t] += 1

        max_freq = int(self.total_records * BLOCKING_CONFIG["max_token_freq"])
        frequent_name_tokens = {t for t, f in self.name_token_freq.items() if f > max_freq}
        frequent_addr_tokens = {t for t, f in self.addr_token_freq.items() if f > max_freq}

        for idx in range(self.total_records):
            core = self.name_cores[idx]
            compact = self.name_compacts[idx]
            first_toks = self.name_first_tokens[idx]
            n_tokens = self.name_tokens[idx]
            a_tokens = self.address_tokens[idx]
            h_nums = self.house_numbers[idx]
            post_code = self.postal_codes[idx]
            st = self.states[idx]

            if core:
                self.exact_core_index[core].append(idx)
            if compact and len(compact) >= 4:
                self.compact_name_index[compact].append(idx)
            if first_toks:
                self.first_tokens_index[first_toks].append(idx)

            for t in set(n_tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"] and t not in frequent_name_tokens:
                    self.rare_name_token_index[t].append(idx)

            if h_nums and n_tokens:
                for num in h_nums[:2]:
                    for t in n_tokens[:2]:
                        if len(t) >= 3:
                            self.name_house_num_index[f"{num}_{t}"].append(idx)

            if h_nums and a_tokens:
                for num in h_nums[:2]:
                    for at in a_tokens:
                        if len(at) >= 4 and at not in frequent_addr_tokens:
                            self.house_addr_token_index[f"{num}_{at}"].append(idx)

            if (post_code or st) and n_tokens:
                geo_key = post_code or st
                for t in n_tokens[:2]:
                    if len(t) >= 3:
                        self.postal_name_index[f"{geo_key}_{t}"].append(idx)

    def _build_tfidf_indexes(self):
        """Build character & word TF-IDF vector matrices (Channels H, I, J)."""
        if self.total_records < 5:
            return

        try:
            cores_text = [c if c else "empty" for c in self.name_cores]
            self.char_tfidf = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=BLOCKING_CONFIG["tfidf_char_ngram_range"],
                max_features=BLOCKING_CONFIG["tfidf_max_features"],
                dtype=np.float32,
            )
            self.char_tfidf_matrix = self.char_tfidf.fit_transform(cores_text)

            addrs_text = [a if a else "empty" for a in self.address_norms]
            self.addr_tfidf = TfidfVectorizer(
                analyzer="word",
                ngram_range=BLOCKING_CONFIG["tfidf_word_ngram_range"],
                max_features=BLOCKING_CONFIG["tfidf_max_features"],
                dtype=np.float32,
            )
            self.addr_tfidf_matrix = self.addr_tfidf.fit_transform(addrs_text)
        except Exception as e:
            logger.warning(f"TF-IDF indexing failed for country {self.country}: {e}")


class MultiChannelBlockingEngine:
    """
    Coordinates multi-channel candidate generation across country partitions.
    """
    def __init__(self, targets_df: pd.DataFrame):
        self.targets_df = targets_df
        self.country_indices: Dict[str, CountryBlockingIndex] = {}
        
        countries = targets_df["country"].unique()
        for country in countries:
            subset = targets_df[targets_df["country"] == country]
            self.country_indices[country] = CountryBlockingIndex(country, subset)
            logger.info(f"Built blocking index for country '{country}': {len(subset):,} target records")

    def retrieve_candidates_for_s1(
        self,
        s1_row: Any,
        top_k: int = TOP_K_CANDIDATES
    ) -> List[Tuple[str, float, int]]:
        """
        Retrieve candidates for a single S1 entity using Channels A through G + fast heuristics.
        """
        country = getattr(s1_row, "country", "unknown")
        
        if country not in self.country_indices:
            if not self.country_indices:
                return []
            country = max(self.country_indices.keys(), key=lambda k: self.country_indices[k].total_records)

        index = self.country_indices[country]
        channel_hits = defaultdict(set)

        # Channel A: exact core name
        core = getattr(s1_row, "name_core", "")
        if core and core in index.exact_core_index:
            for c_idx in index.exact_core_index[core]:
                channel_hits[c_idx].add("ChA_ExactCore")

        # Channel B: compact name
        compact = getattr(s1_row, "name_compact", "")
        if compact and compact in index.compact_name_index:
            for c_idx in index.compact_name_index[compact]:
                channel_hits[c_idx].add("ChB_Compact")

        # Channel C: first 2 core name tokens
        first_toks = getattr(s1_row, "name_first_tokens", "")
        if first_toks and first_toks in index.first_tokens_index:
            for c_idx in index.first_tokens_index[first_toks]:
                channel_hits[c_idx].add("ChC_FirstToks")

        # Channel D: rare name tokens
        n_tokens = getattr(s1_row, "name_tokens", [])
        for t in set(n_tokens):
            if len(t) >= BLOCKING_CONFIG["min_token_len"] and t in index.rare_name_token_index:
                for c_idx in index.rare_name_token_index[t]:
                    channel_hits[c_idx].add("ChD_RareNameTok")

        # Channel E: country + name token + house number
        h_nums = getattr(s1_row, "house_number", [])
        if h_nums and n_tokens:
            for num in h_nums[:2]:
                for t in n_tokens[:2]:
                    key = f"{num}_{t}"
                    if key in index.name_house_num_index:
                        for c_idx in index.name_house_num_index[key]:
                            channel_hits[c_idx].add("ChE_HouseNameTok")

        # Channel F: house number + rare address token
        a_tokens = getattr(s1_row, "address_tokens", [])
        if h_nums and a_tokens:
            for num in h_nums[:2]:
                for at in a_tokens:
                    key = f"{num}_{at}"
                    if key in index.house_addr_token_index:
                        for c_idx in index.house_addr_token_index[key]:
                            channel_hits[c_idx].add("ChF_HouseAddrTok")

        # Channel G: postal code / state + name token
        post_code = getattr(s1_row, "postal_code", "")
        st = getattr(s1_row, "state", "")
        if (post_code or st) and n_tokens:
            geo_key = post_code or st
            for t in n_tokens[:2]:
                key = f"{geo_key}_{t}"
                if key in index.postal_name_index:
                    for c_idx in index.postal_name_index[key]:
                        channel_hits[c_idx].add("ChG_GeoNameTok")

        if not channel_hits:
            return []

        scored_candidates = []
        s1_h_nums = set(h_nums)
        s1_post = post_code
        s1_st = st

        for c_idx, channels in channel_hits.items():
            c_eid = index.entity_ids[c_idx]
            c_core = index.name_cores[c_idx]
            c_compact = index.name_compacts[c_idx]
            c_n_tokens = index.name_tokens[c_idx]
            c_a_tokens = index.address_tokens[c_idx]
            c_h_nums = set(index.house_numbers[c_idx])
            c_post = index.postal_codes[c_idx]
            c_st = index.states[c_idx]

            name_jaccard = compute_quick_jaccard(n_tokens, c_n_tokens)
            addr_jaccard = compute_quick_jaccard(a_tokens, c_a_tokens)
            
            is_exact_core = 1.0 if (core and core == c_core) else 0.0
            is_exact_compact = 1.0 if (compact and compact == c_compact) else 0.0
            num_match = 1.0 if (s1_h_nums and s1_h_nums & c_h_nums) else 0.0
            post_match = 1.0 if (s1_post and s1_post == c_post) else 0.0
            state_match = 1.0 if (s1_st and s1_st == c_st) else 0.0

            channel_support = len(channels)

            total_score = (
                is_exact_core * 3.5
                + is_exact_compact * 2.0
                + name_jaccard * 3.0
                + addr_jaccard * 1.5
                + num_match * 1.0
                + post_match * 1.0
                + state_match * 0.5
                + channel_support * 0.8
            )
            scored_candidates.append((c_eid, total_score, channel_support))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]