"""
Model training, entity-level validation, and threshold optimization pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    VAL_SPLIT_RATIO,
    MODEL_CONFIG,
    THRESHOLDS_GRID,
    RANDOM_SEED,
)
from src.features import extract_features_for_candidates, FEATURE_NAMES
from src.evaluate import evaluate_predictions
from src.utils import logger, time_block, evaluate_macro_f05


def create_labeled_dataset(
    pair_ids: List[Tuple[str, str]],
    X: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]]
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Assign binary labels y (1 if match, 0 if negative) to candidate pairs.
    Returns: X, y, s1_ids_list (aligned with rows)
    """
    y = np.zeros(len(pair_ids), dtype=np.int32)
    s1_ids_list = []

    for i, (s1, cand) in enumerate(pair_ids):
        s1_ids_list.append(s1)
        true_set = ground_truth_dict.get(s1, set())
        if cand in true_set:
            y[i] = 1

    positives = int(np.sum(y))
    negatives = len(y) - positives
    logger.info(f"Dataset labels constructed: {len(y):,} total pairs ({positives:,} positive, {negatives:,} negative | Positive ratio: {positives/len(y):.2%})")

    return X, y, s1_ids_list


@time_block("Threshold Optimization for Macro F0.5")
def find_optimal_threshold(
    val_s1_ids: List[str],
    val_pair_ids: List[Tuple[str, str]],
    val_probs: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]],
    thresholds: List[float] = THRESHOLDS_GRID
) -> Tuple[float, float, Dict[float, float]]:
    """
    Search threshold grid to maximize Macro F0.5 score on validation entities.
    """
    logger.info("Searching for optimal decision threshold maximizing validation Macro F0.5...")
    best_threshold = 0.65
    best_score = -1.0
    threshold_scores = {}

    # Organize predictions per S1 entity
    val_s1_pairs = {}
    for (s1, cand), prob in zip(val_pair_ids, val_probs):
        if s1 not in val_s1_pairs:
            val_s1_pairs[s1] = []
        val_s1_pairs[s1].append((cand, prob))

    for thresh in thresholds:
        pred_dict = {}
        for s1 in val_s1_ids:
            pairs = val_s1_pairs.get(s1, [])
            matched = {cand for cand, prob in pairs if prob >= thresh}
            pred_dict[s1] = matched

        eval_res = evaluate_macro_f05(
            ground_truth=ground_truth_dict,
            predictions=pred_dict,
            s1_ids=val_s1_ids
        )
        score = eval_res["macro_f05"]
        threshold_scores[thresh] = score
        logger.info(f"  Threshold {thresh:.2f} -> Macro F0.5: {score:.4f} (Precision: {eval_res['macro_precision']:.4f}, Recall: {eval_res['macro_recall']:.4f})")

        if score > best_score:
            best_score = score
            best_threshold = thresh

    logger.info(f"★ OPTIMAL THRESHOLD FOUND: {best_threshold:.2f} (Macro F0.5 = {best_score:.4f})")
    return best_threshold, best_score, threshold_scores


@time_block("Model Training Pipeline")
def train_matcher_model(
    candidates_dict: Dict[str, List[Tuple[str, float]]],
    s1_preprocessed: pd.DataFrame,
    targets_preprocessed: pd.DataFrame,
    ground_truth_dict: Dict[str, Set[str]],
    model_type: str = "hist_gb"
) -> Tuple[Any, float, Dict[str, Any]]:
    """
    Extract features, train classifier, optimize threshold, and evaluate on validation set.
    """
    # 1. Entity-level Train / Validation split
    all_s1_ids = list(s1_preprocessed["entity_id"].unique())
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids,
        test_size=VAL_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
    logger.info(f"Entity-level Split: {len(train_s1_ids):,} Train S1 entities, {len(val_s1_ids):,} Validation S1 entities")

    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    # Split candidate dict
    train_cands = {s1: cands for s1, cands in candidates_dict.items() if s1 in train_s1_set}
    val_cands = {s1: cands for s1, cands in candidates_dict.items() if s1 in val_s1_set}

    # 2. Extract pairwise features
    logger.info("Extracting training features...")
    X_train_raw, train_pair_ids = extract_features_for_candidates(
        train_cands, s1_preprocessed, targets_preprocessed
    )
    X_train, y_train, _ = create_labeled_dataset(train_pair_ids, X_train_raw, ground_truth_dict)

    logger.info("Extracting validation features...")
    X_val_raw, val_pair_ids = extract_features_for_candidates(
        val_cands, s1_preprocessed, targets_preprocessed
    )
    X_val, y_val, _ = create_labeled_dataset(val_pair_ids, X_val_raw, ground_truth_dict)

    # 3. Train Classifier
    logger.info(f"Training {model_type} classifier on {len(X_train):,} candidate pairs...")
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=MODEL_CONFIG["n_estimators"],
            max_depth=MODEL_CONFIG["max_depth"],
            min_samples_leaf=MODEL_CONFIG["min_samples_leaf"],
            class_weight=MODEL_CONFIG["class_weight"],
            random_state=RANDOM_SEED,
            n_jobs=MODEL_CONFIG["n_jobs"]
        )
    else:
        # Fast histogram-based gradient boosting
        # Calculate positive class weight for class imbalance
        pos_weight = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
        sample_weights = np.where(y_train == 1, pos_weight, 1.0)
        
        model = HistGradientBoostingClassifier(
            max_iter=150,
            max_depth=8,
            min_samples_leaf=10,
            random_state=RANDOM_SEED,
            scoring="loss"
        )
        model.sample_weights = sample_weights

    if model_type == "rf":
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train, sample_weight=sample_weights)

    # 4. Predict probabilities on validation set
    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.zeros(0)

    # 5. Tune threshold for Macro F0.5
    best_threshold, best_val_score, thresh_scores = find_optimal_threshold(
        val_s1_ids=val_s1_ids,
        val_pair_ids=val_pair_ids,
        val_probs=val_probs,
        ground_truth_dict=ground_truth_dict
    )

    # 6. Full evaluation on validation set with optimal threshold
    val_pred_dict = {}
    for (s1, cand), prob in zip(val_pair_ids, val_probs):
        if s1 not in val_pred_dict:
            val_pred_dict[s1] = set()
        if prob >= best_threshold:
            val_pred_dict[s1].add(cand)

    val_eval = evaluate_predictions(
        ground_truth=ground_truth_dict,
        predictions=val_pred_dict,
        s1_metadata_df=s1_preprocessed,
        s1_ids=val_s1_ids
    )

    # 7. Save model and metadata
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    metadata = {
        "model_type": model_type,
        "feature_names": FEATURE_NAMES,
        "optimal_threshold": float(best_threshold),
        "validation_macro_f05": float(best_val_score),
        "validation_metrics": val_eval,
        "threshold_scores": {str(k): float(v) for k, v in thresh_scores.items()},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {METADATA_PATH}")

    return model, best_threshold, metadata
