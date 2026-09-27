"""
High-Recall Inverted Index and Multi-Strategy Blocking Engine.
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple, Any, Optional
import pandas as pd
from src.config import BLOCKING_CONFIG
from src.utils import logger, time_block


class EntityIndex:
    """
    Multi-key inverted index for fast candidate retrieval within a country partition.
    """
    def __init__(self, target_df: pd.DataFrame):
        self.exact_name_index = defaultdict(list)
        self.alnum_name_index = defaultdict(list)
        self.name_token_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)
        self.addr_num_name_index = defaultdict(list)
        self.name_prefix_index = defaultdict(list)
        self.name_3gram_index = defaultdict(list)
        
        self.token_frequencies = defaultdict(int)
        self.addr_token_frequencies = defaultdict(int)
        self.total_records = len(target_df)

        # Pre-extract native lists for fast index-based scoring (1000x faster than df.iloc / df.at)
        self.entity_ids = target_df["entity_id"].tolist()
        self.name_no_suffixes = target_df["name_no_suffix"].tolist()
        self.name_alnums = target_df["name_alnum"].tolist() if "name_alnum" in target_df.columns else [""] * self.total_records
        self.name_tokens = target_df["name_tokens"].tolist()
        self.address_tokens = target_df["address_tokens"].tolist()
        self.address_numbers = target_df["address_numbers"].tolist()
        
        self._build_indexes()

    def _build_indexes(self):
        """Construct inverted indexes across multiple representations."""
        # 1. Compute name and address token frequencies
        for tokens in self.name_tokens:
            for t in set(tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"]:
                    self.token_frequencies[t] += 1

        for tokens in self.address_tokens:
            for t in set(tokens):
                if len(t) >= 4:
                    self.addr_token_frequencies[t] += 1

        max_freq = int(self.total_records * BLOCKING_CONFIG["max_token_freq"])
        frequent_name_tokens = {
            t for t, freq in self.token_frequencies.items() if freq > max_freq
        }
        frequent_addr_tokens = {
            t for t, freq in self.addr_token_frequencies.items() if freq > max_freq
        }

        # 2. Populate indexes
        for idx in range(self.total_records):
            name_no_sfx = self.name_no_suffixes[idx]
            name_alnum = self.name_alnums[idx]
            tokens = self.name_tokens[idx]
            addr_tokens = self.address_tokens[idx]
            addr_numbers = self.address_numbers[idx]

            # Key 1: Exact clean name
            if name_no_sfx:
                self.exact_name_index[name_no_sfx].append(idx)

            # Key 2: Alphanumeric compact name
            if name_alnum and len(name_alnum) >= 4:
                self.alnum_name_index[name_alnum].append(idx)

            # Key 3: Distinctive name tokens
            for t in set(tokens):
                if len(t) >= BLOCKING_CONFIG["min_token_len"] and t not in frequent_name_tokens:
                    self.name_token_index[t].append(idx)

            # Key 4: Distinctive address tokens
            for t in set(addr_tokens):
                if len(t) >= 4 and t not in frequent_addr_tokens:
                    self.addr_token_index[t].append(idx)

            # Key 5: Name 2-token prefix key
            if len(tokens) >= 2:
                prefix_key = f"{tokens[0]}_{tokens[1]}"
                self.name_prefix_index[prefix_key].append(idx)
            elif len(tokens) == 1 and len(tokens[0]) >= 4:
                self.name_prefix_index[tokens[0]].append(idx)

            # Key 6: Address number + First name token
            if addr_numbers and tokens:
                first_tok = tokens[0]
                if len(first_tok) >= 3:
                    for num in addr_numbers[:2]:
                        num_key = f"{num}_{first_tok}"
                        self.addr_num_name_index[num_key].append(idx)

            # Key 7: Name 3-grams for typo resilience
            if name_alnum and len(name_alnum) >= 6:
                for i in range(len(name_alnum) - 2):
                    g = name_alnum[i:i+3]
                    if self.token_frequencies.get(g, 0) <= max_freq:
                        self.name_3gram_index[g].append(idx)


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


class BlockingEngine:
    """
    Coordinates blocking across country partitions and sources.
    """
    def __init__(self, target_df: pd.DataFrame):
        """
        target_df contains combined Source 2 and Source 3 preprocessed records.
        """
        self.target_df = target_df
        # Partition target records by country
        self.country_indices = {}
        
        countries = target_df["country"].unique()
        for country in countries:
            subset = target_df[target_df["country"] == country].reset_index(drop=True)
            self.country_indices[country] = EntityIndex(subset)
            logger.info(f"Built blocking index for country '{country}': {len(subset):,} records")

    def get_candidates_for_record(
        self,
        s1_row: Any,
        max_candidates: int = 35
    ) -> List[Tuple[str, float]]:
        """
        Retrieve candidate entity IDs for a single Source 1 record.
        Returns list of (candidate_entity_id, preliminary_score).
        """
        country = s1_row.country
        if country not in self.country_indices:
            # Open-set country with no targets in that country
            return []

        index = self.country_indices[country]
        candidate_indices = set()

        # Strategy 1: Exact name no suffix
        name_no_sfx = s1_row.name_no_suffix
        if name_no_sfx:
            candidate_indices.update(index.exact_name_index.get(name_no_sfx, []))

        # Strategy 2: Alphanumeric name
        name_alnum = s1_row.name_alnum
        if name_alnum and len(name_alnum) >= 4:
            candidate_indices.update(index.alnum_name_index.get(name_alnum, []))

        # Strategy 3: 2-token prefix key
        tokens = s1_row.name_tokens
        if len(tokens) >= 2:
            prefix_key = f"{tokens[0]}_{tokens[1]}"
            candidate_indices.update(index.name_prefix_index.get(prefix_key, []))
        elif len(tokens) == 1 and len(tokens[0]) >= 4:
            candidate_indices.update(index.name_prefix_index.get(tokens[0], []))

        # Strategy 4: Distinctive name token overlap
        for t in set(tokens):
            if len(t) >= BLOCKING_CONFIG["min_token_len"]:
                candidate_indices.update(index.name_token_index.get(t, []))

        # Strategy 5: Address number + First name token
        addr_numbers = s1_row.address_numbers
        if addr_numbers and tokens:
            first_tok = tokens[0]
            if len(first_tok) >= 3:
                for num in addr_numbers[:2]:
                    num_key = f"{num}_{first_tok}"
                    candidate_indices.update(index.addr_num_name_index.get(num_key, []))

        # Strategy 6: Distinctive address token matches
        addr_tokens = s1_row.address_tokens
        for at in set(addr_tokens):
            if len(at) >= 5:
                # Add up to 50 targets per distinctive address token
                cands = index.addr_token_index.get(at, [])
                if len(cands) <= 50:
                    candidate_indices.update(cands)

        if not candidate_indices:
            return []

        # Rank and score candidates using composite similarity
        scored_candidates = []
        s1_tokens = s1_row.name_tokens
        s1_addr_tokens = s1_row.address_tokens
        s1_nums = set(addr_numbers)

        for c_idx in candidate_indices:
            c_eid = index.entity_ids[c_idx]
            c_name_no_sfx = index.name_no_suffixes[c_idx]
            c_name_tokens = index.name_tokens[c_idx]
            c_addr_tokens = index.address_tokens[c_idx]
            c_nums = set(index.address_numbers[c_idx])

            name_jaccard = compute_quick_jaccard(s1_tokens, c_name_tokens)
            addr_jaccard = compute_quick_jaccard(s1_addr_tokens, c_addr_tokens)
            num_overlap = 1.0 if (s1_nums and s1_nums & c_nums) else 0.0
            
            is_exact = 1.0 if (name_no_sfx and name_no_sfx == c_name_no_sfx) else 0.0
            
            total_score = (
                is_exact * 2.5
                + name_jaccard * 2.0
                + addr_jaccard * 1.5
                + num_overlap * 0.8
            )
            scored_candidates.append((c_eid, total_score))

        # Sort descending by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates[:max_candidates]