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
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
OUTPUT_EDA_REPORT_PATH = OUTPUT_DIR / "eda_report.txt"

# Model paths
MODEL_PATH = MODELS_DIR / "entity_matcher.pkl"
METADATA_PATH = MODELS_DIR / "matcher_metadata.json"

# Reproducibility
RANDOM_SEED = 42

# Pipeline Parameters
BLOCKING_CONFIG = {
    "min_token_len": 3,
    "max_token_freq": 0.05,  # Filter top 5% most frequent tokens as stopwords
    "max_candidates_per_s1": 25,  # Keep top candidates per S1 to ensure lean candidate set
    "min_jaccard_filter": 0.15,  # Min token jaccard to retain fuzzy candidate
}

# Validation Split
VAL_SPLIT_RATIO = 0.20

# Model Hyperparameters
MODEL_CONFIG = {
    "n_estimators": 250,
    "max_depth": 10,
    "min_samples_leaf": 4,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Threshold tuning grid for Macro F0.5 optimization
THRESHOLDS_GRID = [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
DEFAULT_DECISION_THRESHOLD = 0.65
