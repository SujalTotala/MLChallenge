"""
Multi-Channel Scalable Blocking Engine for Business Entity Resolution.
Implements Channels A through L with Inverted Indexes and Fast Vectorized Candidate Retrieval.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import BLOCKING_CONFIG, TOP_K_CANDIDATES
from src.utils import logger, time_block


class MultiChannelBlockingEngine:
    """
    Coordinates multi-channel candidate generation (Channels A - L) across target databases (S2 + S3).
    Global index retrieval avoids cross-country data loss while retaining country match scoring.
    """
    def __init__(self, targets_df: pd.DataFrame):
        self.targets_df = targets_df.reset_index(drop=True)
        self.total_records = len(self.targets_df)

        self.entity_ids = self.targets_df["entity_id"].tolist()
        self.name_cores = self.targets_df["name_core"].tolist()
        self.name_compacts = self.targets_df["name_compact"].tolist()
        self.name_tokens = self.targets_df["name_tokens"].tolist()
        self.name_first_tokens = self.targets_df["name_first_tokens"].tolist()
        self.address_norms = self.targets_df["address_norm"].tolist()
        self.address_tokens = self.targets_df["address_tokens"].tolist()
        self.house_numbers = self.targets_df["house_number"].tolist()
        self.postal_codes = self.targets_df["postal_code"].tolist()
        self.states = self.targets_df["state"].tolist()
        self.countries = self.targets_df["country"].tolist()
        self.address_compacts = self.targets_df["address_compact"].tolist()

        # Pre-convert token lists to sets for sub-microsecond set operations
        self.name_tokens_sets = [set(t) for t in self.name_tokens]
        self.addr_tokens_sets = [set(t) for t in self.address_tokens]
        self.house_numbers_sets = [set(h) for h in self.house_numbers]

        # Channels A - H Inverted Indexes
        self.exact_core_index = defaultdict(list)       # Channel A
        self.compact_name_index = defaultdict(list)     # Channel B
        self.rare_name_token_index = defaultdict(list)  # Channel C
        self.prefix_index = defaultdict(list)           # Channel D
        self.name_house_num_index = defaultdict(list)   # Channel E
        self.house_addr_token_index = defaultdict(list) # Channel F
        self.postal_name_index = defaultdict(list)      # Channel G
        self.addr_compact_index = defaultdict(list)     # Channel H

        self.name_token_freq = defaultdict(int)
        self.addr_token_freq = defaultdict(int)

        self._build_indexes()

    def _build_indexes(self):
        """Build channels A through H inverted indexes across target records."""
        logger.info(f"Building multi-channel inverted indexes for {self.total_records:,} target records...")
        
        for tokens in self.name_tokens:
            for t in set(tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"]:
                    self.name_token_freq[t] += 1

        for tokens in self.address_tokens:
            for t in set(tokens):
                if len(t) >= 3:
                    self.addr_token_freq[t] += 1

        max_freq = int(self.total_records * BLOCKING_CONFIG["max_token_freq"])
        frequent_name_tokens = {t for t, f in self.name_token_freq.items() if f > max_freq}

        for idx in range(self.total_records):
            core = self.name_cores[idx]
            compact = self.name_compacts[idx]
            first_toks = self.name_first_tokens[idx]
            n_tokens = self.name_tokens[idx]
            a_tokens = self.address_tokens[idx]
            h_nums = self.house_numbers[idx]
            post_code = self.postal_codes[idx]
            st = self.states[idx]
            a_compact = self.address_compacts[idx]

            # Channel A: exact core
            if core:
                self.exact_core_index[core].append(idx)
            # Channel B: compact name
            if compact and len(compact) >= 3:
                self.compact_name_index[compact].append(idx)
            # Channel D: prefix / 3-gram
            if first_toks:
                self.prefix_index[first_toks].append(idx)
            for t in n_tokens:
                if len(t) >= 3:
                    self.prefix_index[t[:3]].append(idx)

            # Channel C: rare tokens
            for t in set(n_tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"] and t not in frequent_name_tokens:
                    self.rare_name_token_index[t].append(idx)

            # Channel E: name token + house number
            if h_nums and n_tokens:
                for num in h_nums[:2]:
                    for t in n_tokens[:3]:
                        if len(t) >= 3:
                            self.name_house_num_index[f"{num}_{t}"].append(idx)

            # Channel F: house number + address token
            if h_nums and a_tokens:
                for num in h_nums[:2]:
                    for at in a_tokens[:3]:
                        if len(at) >= 3 and not at.isdigit():
                            self.house_addr_token_index[f"{num}_{at}"].append(idx)

            # Channel G: postal code / state + name token
            if (post_code or st) and n_tokens:
                geo_key = post_code or st
                for t in n_tokens[:3]:
                    if len(t) >= 3:
                        self.postal_name_index[f"{geo_key}_{t}"].append(idx)

            # Channel H: address compact
            if a_compact and len(a_compact) >= 6:
                self.addr_compact_index[a_compact[:8]].append(idx)

    def retrieve_candidates_for_s1(
        self,
        s1_row: Any,
        top_k: int = TOP_K_CANDIDATES
    ) -> List[Tuple[str, float, int]]:
        """
        Retrieve candidates for a single S1 entity using Channels A through H with ultra-fast candidate scoring.
        """
        channel_hits = defaultdict(int)

        core = getattr(s1_row, "name_core", "")
        compact = getattr(s1_row, "name_compact", "")
        first_toks = getattr(s1_row, "name_first_tokens", "")
        n_tokens = getattr(s1_row, "name_tokens", [])
        h_nums = getattr(s1_row, "house_number", [])
        a_tokens = getattr(s1_row, "address_tokens", [])
        post_code = getattr(s1_row, "postal_code", "")
        st = getattr(s1_row, "state", "")
        country = getattr(s1_row, "country", "unknown")
        a_compact = getattr(s1_row, "address_compact", "")

        s1_n_set = set(n_tokens)
        s1_a_set = set(a_tokens)
        s1_h_set = set(h_nums)

        # Channel A: exact core name
        if core and core in self.exact_core_index:
            for c_idx in self.exact_core_index[core]:
                channel_hits[c_idx] += 4

        # Channel B: compact name
        if compact and compact in self.compact_name_index:
            for c_idx in self.compact_name_index[compact]:
                channel_hits[c_idx] += 3

        # Channel C: rare name tokens
        for t in s1_n_set:
            if len(t) >= BLOCKING_CONFIG["min_token_len"] and t in self.rare_name_token_index:
                for c_idx in self.rare_name_token_index[t]:
                    channel_hits[c_idx] += 2

        # Channel D: name prefixes & 3-gram tokens
        if first_toks and first_toks in self.prefix_index:
            for c_idx in self.prefix_index[first_toks]:
                channel_hits[c_idx] += 2
        for t in n_tokens:
            if len(t) >= 3:
                p3 = t[:3]
                if p3 in self.prefix_index:
                    for c_idx in self.prefix_index[p3]:
                        channel_hits[c_idx] += 1

        # Channel E: name token + house number
        if h_nums and n_tokens:
            for num in h_nums[:2]:
                for t in n_tokens[:3]:
                    key = f"{num}_{t}"
                    if key in self.name_house_num_index:
                        for c_idx in self.name_house_num_index[key]:
                            channel_hits[c_idx] += 3

        # Channel F: house number + address token
        if h_nums and a_tokens:
            for num in h_nums[:2]:
                for at in a_tokens[:3]:
                    if len(at) >= 3 and not at.isdigit():
                        key = f"{num}_{at}"
                        if key in self.house_addr_token_index:
                            for c_idx in self.house_addr_token_index[key]:
                                channel_hits[c_idx] += 2

        # Channel G: postal code / state + name token
        if (post_code or st) and n_tokens:
            geo_key = post_code or st
            for t in n_tokens[:3]:
                if len(t) >= 3:
                    key = f"{geo_key}_{t}"
                    if key in self.postal_name_index:
                        for c_idx in self.postal_name_index[key]:
                            channel_hits[c_idx] += 2

        # Channel H: address compact
        if a_compact and len(a_compact) >= 6:
            key = a_compact[:8]
            if key in self.addr_compact_index:
                for c_idx in self.addr_compact_index[key]:
                    channel_hits[c_idx] += 2

        if not channel_hits:
            return []

        scored_candidates = []

        for c_idx, ch_score in channel_hits.items():
            c_eid = self.entity_ids[c_idx]
            c_core = self.name_cores[c_idx]
            c_compact = self.name_compacts[c_idx]
            c_n_set = self.name_tokens_sets[c_idx]
            c_a_set = self.addr_tokens_sets[c_idx]
            c_h_set = self.house_numbers_sets[c_idx]
            c_post = self.postal_codes[c_idx]
            c_cntry = self.countries[c_idx]

            # Fast Jaccard overlap
            n_inter = len(s1_n_set & c_n_set)
            n_jacc = (n_inter / len(s1_n_set | c_n_set)) if (s1_n_set and c_n_set) else 0.0

            a_inter = len(s1_a_set & c_a_set)
            a_jacc = (a_inter / len(s1_a_set | c_a_set)) if (s1_a_set and c_a_set) else 0.0

            is_exact_core = 4.0 if (core and core == c_core) else 0.0
            is_exact_compact = 3.0 if (compact and compact == c_compact) else 0.0
            num_match = 1.5 if (s1_h_set and s1_h_set & c_h_set) else 0.0
            post_match = 1.0 if (post_code and post_code == c_post) else 0.0
            country_match = 0.5 if (country and country == c_cntry) else 0.0

            total_score = (
                is_exact_core
                + is_exact_compact
                + n_jacc * 3.5
                + a_jacc * 1.5
                + num_match
                + post_match
                + country_match
                + ch_score * 0.5
            )
            scored_candidates.append((c_eid, total_score, ch_score))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:top_k]