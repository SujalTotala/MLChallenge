# Amazon ML Challenge 2026: Business Entity Resolution

High-performance, scalable entity resolution pipeline built for the Amazon ML Challenge 2026. Strictly compliant with official challenge rules, open-set country distribution, and Macro F0.5 optimization.

---

## Key Features & Architecture

```
RAW DATA
  ↓
MULTI-REPRESENTATION NORMALIZATION
  ↓
MULTI-CHANNEL BLOCKING (Channels A - K)
  ↓
CANDIDATE RANKING (TOP-K = 25)
  ↓
candidate_pairs.tsv
  ↓
45-DIMENSIONAL PAIRWISE FEATURE ENGINEERING
  ↓
LIGHTGBM MATCH CLASSIFIER
  ↓
SINGLETON-AWARE F0.5 THRESHOLD OPTIMIZATION
  ↓
matching_results.tsv
  ↓
OFFICIAL VALIDATOR (PASS)
```

1. **Multi-Representation Text Normalization**:
   - `name_raw`, `name_norm`, `name_core` (legal suffixes stripped), `name_compact` (alphanumeric), `name_tokens`, `name_sorted`, `name_first_tokens`.
   - `address_norm` (abbreviations expanded), `address_tokens`, `house_number`, `postal_code` (IN PIN / US ZIP / FR Code Postal), `state`, `street_tokens`.
   - Open-set country normalization (`country_norm`) supporting US, India, France, and any unseen future country.

2. **Multi-Channel Blocking Engine (Channels A through K)**:
   - **Channel A**: `country + exact normalized core name`
   - **Channel B**: `country + compact name`
   - **Channel C**: `country + first 2 core name tokens`
   - **Channel D**: `country + rare name token` (filtering top frequency stopwords)
   - **Channel E**: `country + name token + house number`
   - **Channel F**: `country + house number + rare address token`
   - **Channel G**: `country + postal code / state + name token`
   - **Channel H**: Character TF-IDF (3-5 char ngrams) sparse nearest-neighbor retrieval on core name
   - **Channel I**: Word TF-IDF (1-2 word ngrams) sparse nearest-neighbor retrieval on address
   - **Channel J**: Combined name + address TF-IDF retrieval
   - **Channel K**: Reverse retrieval (S1 → S2/S3 AND S2/S3 → S1)

3. **High-Precision Machine Learning & Optimization**:
   - **LightGBM Classifier**: Efficient, MIT-licensed gradient boosting tree model with positive class imbalance weighting.
   - **45 Pairwise Features**: RapidFuzz string metrics, TF-IDF cosine similarities, geo-conflict indicators (pincode/house number conflicts), and interaction terms.
   - **Singleton-Aware Decision Logic**: Optimized threshold and probability margin rules to prevent heavy false-positive penalties on singleton entities.

---

## Instructions & Quick Start

### Installation

Ensure required dependencies are installed:
```bash
pip install -r requirements.txt
```

### Fast Development Mode (--mode sample)

Run quick pipeline on a sampled dataset (fast validation):
```bash
python run_pipeline.py --mode sample
```

### Full Production Pipeline (--mode full)

Run complete entity resolution pipeline across ALL dataset records:
```bash
python run_pipeline.py --mode full
```

### Running Individual Stages

```bash
python run_pipeline.py --stage train --mode sample
python run_pipeline.py --stage predict --mode full
python run_pipeline.py --stage validate
```

---

## Submission Output Validation

Validate output compliance with the official submission validator:
```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

---

## Project Structure

```
├── dataset/
│   ├── train/ (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/  (test_source1.tsv, test_source2.tsv, test_source3.tsv)
├── src/
│   ├── config.py             # Config & Hyperparameters
│   ├── normalize.py          # Multi-representation Normalization
│   ├── data_loader.py        # Ingestion & Quality Metrics
│   ├── blocking.py           # Multi-Channel Blocking Engine (A-K)
│   ├── candidate_generation.py # Candidate Pairs Export & Recall Evaluation
│   ├── hard_negatives.py     # Hard Negative Sampling
│   ├── features.py           # 45-dimensional Pairwise Features
│   ├── train_model.py        # LightGBM Training & Threshold Tuning
│   ├── evaluate.py           # Evaluation & Diagnostic Reports
│   ├── predict.py            # Test Inference & Output Generation
│   ├── eda.py                # Exploratory Data Analysis
│   └── utils.py              # Metrics & File I/O Helpers
├── output/
│   ├── matching_results.tsv  # Final predicted matches
│   └── candidate_pairs.tsv   # Candidate pairs set
├── reports/
│   ├── blocking_report.txt   # Detailed candidate recall metrics
│   ├── validation_report.txt # Official Macro F0.5 metrics
│   └── error_analysis.csv    # Worst validation error cases
├── models/
│   ├── entity_matcher.pkl    # Saved LightGBM classifier
│   └── matcher_metadata.json # Metadata, threshold & feature list
├── utils/
│   └── validate_submission.py# Official Submission Validator
├── run_pipeline.py           # Master CLI runner
├── requirements.txt
└── README.md
```
