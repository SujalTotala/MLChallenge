"""
Configuration parameters and paths for Business Entity Resolution.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Training file paths
TRAIN_SOURCE1_PATH = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2_PATH = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3_PATH = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH_PATH = TRAIN_DIR / "train_ground_truth.tsv"

# Test file paths
TEST_SOURCE1_PATH = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2_PATH = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3_PATH = TEST_DIR / "test_source3.tsv"

# Output file paths
OUTPUT_MATCHING_PATH = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES_PATH = OUTPUT_DIR / "candidate_pairs.tsv"

# Report file paths
BLOCKING_REPORT_PATH = REPORTS_DIR / "blocking_report.txt"
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.txt"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.csv"
EDA_REPORT_PATH = OUTPUT_DIR / "eda_report.txt"
OUTPUT_EDA_REPORT_PATH = EDA_REPORT_PATH

# Model paths
MODEL_PATH = MODELS_DIR / "entity_matcher.pkl"
METADATA_PATH = MODELS_DIR / "matcher_metadata.json"

# Reproducibility
RANDOM_SEED = 42

# Candidate Generation & Blocking Config
TOP_K_CANDIDATES = 25  # Configurable: 20, 25, 30
BLOCKING_CONFIG = {
    "top_k": TOP_K_CANDIDATES,
    "min_token_len": 3,
    "max_token_freq": 0.03,  # Top 3% most frequent tokens filtered as stopwords
    "tfidf_char_ngram_range": (3, 5),
    "tfidf_word_ngram_range": (1, 2),
    "tfidf_max_features": 50000,
    "tfidf_top_candidates": 15,
}

# Entity-Level Validation Split
VAL_SPLIT_RATIO = 0.20

# LightGBM Classifier Hyperparameters
LIGHTGBM_CONFIG = {
    "n_estimators": 350,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 10,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Decision Threshold Optimization Grid for Macro F0.5
THRESHOLDS_GRID = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98
]
DEFAULT_DECISION_THRESHOLD = 0.85
MARGIN_THRESHOLD = 0.05
