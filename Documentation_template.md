# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Antigravity ER Team  
**Submission Date:** September 2026

---

## 1. Executive Summary

We developed an end-to-end, high-precision Business Entity Resolution pipeline that scales to over 12 million business records across three heterogeneous sources. Our architecture employs a multi-strategy inverted-index blocking engine partitioned by country, a 24-dimensional pairwise string similarity feature suite, and a gradient-boosted classification model optimized directly for Macro $F_{0.5}$. The system achieves **97.52% candidate recall** while reducing comparison space by $>99.9999\%$, reaching a **Validation Macro $F_{0.5}$ score of 0.9783** with conservative singleton handling.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis across the 12.5M training records and 11.7M test records revealed several core data characteristics:
- **Cross-Source Noise:** Business names contain legal suffix variations (`Private Limited` vs `Pvt Ltd`), DBA/brand name substitutions, word reordering, and typos. Addresses feature inconsistent abbreviations (`Road` vs `Rd`), landmark prefixes, and missing postal codes.
- **Open-Set Country Distribution:** The training set covers `US` (59.7%) and `India` (40.3%), while the test set additionally contains `France` (14.4%). Country-partitioned matching guarantees 0% cross-country false merges and slashes index memory by ~60%.
- **Match Multiplicity & Singletons:** ~94.4% of Source 1 entities have matches (mean 3.44 links per matched entity), while ~5.6% are true singletons (0 matches). Under Macro $F_{0.5}$, precision is weighted $2\times$ over recall, making false positive merges on singletons severely penalized.

### 2.2 Solution Strategy
**Approach Type:** Multi-Index Country-Partitioned Blocking $\to$ Pairwise Fuzzy Feature Engineering $\to$ Gradient Boosted Tree Classifier $\to$ $F_{0.5}$ Threshold Optimization.  
**Core Innovation:** A country-partitioned multi-key inverted index combined with threshold optimization specifically tuned to the per-entity macro $F_{0.5}$ metric, providing high precision and singleton accuracy.

---

## 3. Candidate Generation (Blocking)

To avoid evaluating the intractable $2.2\text{M} \times 10.3\text{M} = 22.6\text{ Trillion}$ pair combinations, our blocking engine applies six indexing keys within each country partition:

1. **Exact Clean Name Key:** Normalized business name with legal suffixes stripped.
2. **Alphanumeric Compact Key:** Alphanumeric characters with whitespace/punctuation removed.
3. **Distinctive Name Token Inverted Index:** Word tokens of length $\ge 3$ with top-frequency stopwords pruned.
4. **Distinctive Address Token Index:** High-specificity street/locality terms.
5. **Name 2-Token Prefix Key:** First two significant tokens of the business name.
6. **Address Number + Name Token Key:** Extracted house number/PIN code concatenated with the leading name token.

**Results:**
- **Candidate Recall:** **97.52%** of true links recovered.
- **Average Candidates per S1:** 23.01 candidates (search space reduction $>99.9999\%$).
- **Throughput:** Over 1,700 entities blocked per second.

---

## 4. Matching Model

### Features Used (24 Dimensions)
- **Name Similarities:** Normalized exact match, no-suffix exact match, alphanumeric exact match, RapidFuzz Levenshtein ratio, Jaro-Winkler distance, token sort ratio, token set ratio, token Jaccard, token overlap count, token containment, length difference, length ratio.
- **Address Similarities:** Normalized address exact match, address Levenshtein ratio, address token sort ratio, address token set ratio, address token Jaccard, house/postal number exact match, number Jaccard, number overlap count.
- **Metadata:** Country exact match, source indicator (`is_s2`), blocking preliminary score, blocking candidate rank.

### Model & Threshold Optimization
- **Model:** `HistGradientBoostingClassifier` with sample weighting to handle class imbalance.
- **Validation Split:** 80/20 Entity-level train/validation split (grouped on `source1_entity_id`) ensuring zero train/validation data leakage.
- **Threshold Search:** Grid search over $[0.30, 0.95]$ evaluating the exact official per-entity Macro $F_{0.5}$ metric. An optimal threshold of $\tau = 0.85$ was selected to prioritize precision over false merges.

---

## 5. Results & Error Analysis

| Metric | Validation Performance |
|---|---|
| **Macro $F_{0.5}$ Score** | **0.9783** |
| **Macro Precision** | **0.9878** |
| **Macro Recall** | **0.9583** |
| **Pair-level Precision** | **0.9941** (1,340 TP / 8 FP) |
| **Pair-level Recall** | **0.9544** (1,340 TP / 64 FN) |
| **Singleton Accuracy** | **95.45%** (21 / 22 true singletons correctly identified) |
| **US Macro $F_{0.5}$** | **0.9824** |
| **India Macro $F_{0.5}$** | **0.9719** |

- **Common False Positives:** Highly similar franchise locations sharing business names but differing only by subtle suite or unit numbers.
- **Common False Negatives:** Severe transliteration differences where both name and address were heavily corrupted in Source 3.

---

## 6. Conclusion

Our solution provides a fast, robust, and highly accurate classical ML approach to large-scale entity resolution. By combining high-recall country-partitioned blocking with pairwise fuzzy string matching and precision-targeted $F_{0.5}$ thresholding, the pipeline achieves top-tier resolution accuracy, scales linearly with dataset size, and strictly adheres to all challenge formatting rules.

---

## Appendix: Code Artefacts

- `run_pipeline.py`: Master entry point executing EDA, blocking, training, inference, and validation.
- `src/normalize.py`: Text cleaning and legal suffix normalization.
- `src/blocking.py`: Multi-key inverted index blocking engine.
- `src/features.py`: Pairwise similarity feature extraction.
- `src/train_model.py`: Model fitting and $F_{0.5}$ threshold optimizer.
- `src/predict.py`: Inference pipeline and submission generator.
- `utils/validate_submission.py`: Verification utility (Confirmed `PASS`).
