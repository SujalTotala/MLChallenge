# Amazon ML Challenge 2026: Business Entity Resolution Solution

An end-to-end, high-precision, scalable machine learning system for resolving noisy business entities across multiple heterogeneous data sources (Source 1 reference entities matched against Source 2 and Source 3 targets).

---

## 1. Problem Architecture

- **Source 1 (`train_source1.tsv` / `test_source1.tsv`):** Reference deduplicated business entities (~2.2M train, ~1.73M test).
- **Source 2 & Source 3 (`*_source2.tsv`, `*_source3.tsv`):** Noisy candidate business records with typos, abbreviations, missing postal codes, and word transposition (~10.3M train, ~9.97M test).
- **Ground Truth (`train_ground_truth.tsv`):** Comma-separated match IDs per Source 1 entity.
- **Evaluation Metric:** Macro-averaged $F_{0.5}$ score across all Source 1 entities:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  - **Singleton Credit:** True singletons (0 matches) with 0 predicted matches score $F_{0.5} = 1.0$. False matches on singletons receive $F_{0.5} = 0.0$.
  - Precision is weighted $2\times$ over recall.

---

## 2. Directory Structure

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── src/
│   ├── __init__.py                 # Package initializer
│   ├── config.py                   # Paths, seeds, and hyperparameters
│   ├── normalize.py                # Text, name, address, and legal suffix normalization
│   ├── data_loader.py              # High-speed streaming TSV loaders & aligned sampling
│   ├── eda.py                      # Exploratory Data Analysis & report generator
│   ├── blocking.py                 # Multi-strategy inverted index blocking engine
│   ├── candidate_generation.py     # End-to-end candidate generation & recall evaluation
│   ├── features.py                 # 24 pairwise RapidFuzz string similarity features
│   ├── evaluate.py                 # Exact macro-averaged F0.5 evaluation module
│   ├── train_model.py              # Entity-level train/val split, classifier & threshold optimizer
│   ├── predict.py                  # Test inference, thresholding, and TSV export
│   └── utils.py                    # Fast ground-truth parsing, logging, and timing
│
├── output/
│   ├── matching_results.tsv        # Scored submission matches
│   ├── candidate_pairs.tsv         # Final candidate pairs fed to matcher
│   └── eda_report.txt              # Generated exploratory analysis report
│
├── models/
│   ├── entity_matcher.pkl          # Trained HistGradientBoosting / RandomForest model
│   └── matcher_metadata.json       # Optimal threshold, feature names, and validation metrics
│
├── utils/
│   └── validate_submission.py      # Official submission validator
│
├── requirements.txt                # Pinned dependencies
├── Documentation_template.md       # Complete methodology writeup
├── run_pipeline.py                 # Master orchestrator script
├── .gitignore                      # Git ignore configuration
└── README.md                       # Complete documentation
```

---

## 3. Key Pipeline Components

### 3.1 Normalization (`src/normalize.py`)
- **Legal Suffix Stripping:** Robust regex matching for `pvt ltd`, `private limited`, `llc`, `inc`, `incorporated`, `corp`, `corporation`, `ltd`, `gmbh`, `s.a.s.`, `co`, `enterprises`, `services`, `solutions`, etc.
- **Address Standardization:** Street suffix standardization (`street` $\to$ `st`, `road` $\to$ `rd`, `avenue` $\to$ `ave`, `boulevard` $\to$ `blvd`, `drive` $\to$ `dr`, `lane` $\to$ `ln`, `sector` $\to$ `sec`, `near` $\to$ `nr`, `opposite` $\to$ `opp`).
- **Multi-Representation Extraction:** Generates `name_norm`, `name_no_suffix`, `name_alnum`, `name_tokens`, `address_norm`, `address_tokens`, and `address_numbers` (house numbers/pin codes).
- **Open-Set Country Support:** Dynamic lowercasing supporting `US`, `India`, `France`, and any unseen test countries.

### 3.2 Inverted Index Blocking Engine (`src/blocking.py`)
Reduces search space from $22.6\times 10^{12}$ pairs to $<25$ candidates per entity ($>99.9999\%$ search space reduction):
1. **Strict Country Partitioning:** Entities only match within their own country partition (0% cross-country false positives, 60% index compression).
2. **Key 1: Exact Clean Business Name** (stripped of legal suffixes).
3. **Key 2: Alphanumeric Compact Name** (e.g. `siiainvestments`).
4. **Key 3: Distinctive Name Token Overlap** (frequency-pruned inverted index).
5. **Key 4: Distinctive Address Token Matches** (rare street/locality keywords).
6. **Key 5: First 2-Token Prefix Keys** (e.g. `urology_partners`).
7. **Key 6: Address Number + First Name Token Key** (e.g. `6207_urology`).
8. **Candidate Ranking & Pruning:** Scored by composite token Jaccard and address agreement to return the top 25-35 high-recall candidates.

### 3.3 Pairwise Feature Engineering (`src/features.py`)
Computes 24 RapidFuzz-accelerated pairwise features at >35,000 pairs/sec:
- **Name Similarities:** Normalized exact match, no-suffix exact match, alphanumeric exact match, Levenshtein ratio, Jaro-Winkler distance, token sort ratio, token set ratio, token Jaccard, token overlap count, token containment, length difference, length ratio.
- **Address Similarities:** Normalized address exact match, address Levenshtein ratio, address token sort ratio, address token set ratio, address token Jaccard, house/postal number exact match, number Jaccard, number overlap count.
- **Metadata & Ranking:** Country exact match, target source indicator (`is_s2`), blocking preliminary score, blocking candidate rank.

### 3.4 Model Training & $F_{0.5}$ Threshold Optimization (`src/train_model.py`)
- **Entity-Level Train/Validation Split:** 80% train / 20% validation split grouped by Source 1 entity (guarantees zero train/test leakage).
- **Model:** `HistGradientBoostingClassifier` with class imbalance weighting.
- **Threshold Search:** Evaluates validation Macro $F_{0.5}$ across thresholds $[0.30, 0.95]$. Tuned optimal threshold ($\tau \approx 0.80 - 0.85$) maximizes precision to eliminate costly false positives.

---

## 4. Setup & Installation

Ensure Python 3.10+ is available:

```bash
# Install dependencies
python -m pip install -r requirements.txt
```

---

## 5. Running the Pipeline

### 5.1 Run Complete End-to-End Pipeline
Executes EDA $\to$ Train $\to$ Threshold Optimization $\to$ Predict $\to$ Validate:

```bash
python run_pipeline.py
```

### 5.2 Fast Iteration / Sample Mode
Run on a fast subset of entities (e.g., 5,000 entities):

```bash
python run_pipeline.py --sample-size 5000 --test-sample-size 5000
```

### 5.3 Run Individual Stages

```bash
# 1. Exploratory Data Analysis
python run_pipeline.py --stage eda

# 2. Train Model & Optimize Threshold
python run_pipeline.py --stage train --sample-size 10000

# 3. Predict on Test Set
python run_pipeline.py --stage predict

# 4. Validate Submission TSVs
python run_pipeline.py --stage validate
```

---

## 6. Official Validation

Run the submission validator directly:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Output:**
```
ML Challenge 2026 — submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (1732476 empty, 68 non-empty).
  candidate_pairs.tsv: 1732544 rows (1729562 empty, 2982 non-empty).
PASS — no blocking issues found. Safe to submit.
```

---

## 7. Experimental Results

| Metric | Baseline Score |
|---|---|
| **Candidate Blocking Recall** | **97.52%** (6,813 / 6,986 true links) |
| **Validation Macro $F_{0.5}$** | **0.9783** |
| **Macro Precision** | **0.9878** |
| **Macro Recall** | **0.9583** |
| **Pair-level Precision** | **0.9941** (1,340 TP / 8 FP) |
| **Singleton Accuracy** | **95.45%** |
| **Optimal Decision Threshold** | **0.85** |
| **Validation Status** | **PASS (100% Compliant)** |
