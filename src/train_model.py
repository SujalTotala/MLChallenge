"""
Model training, entity-level validation, threshold optimization, and singleton-aware decision tuning.
Uses MIT-licensed LightGBM classifier for scalable, high-precision entity resolution.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

from src.config import (
    MODEL_PATH,
    METADATA_PATH,
    VAL_SPLIT_RATIO,
    LIGHTGBM_CONFIG,
    THRESHOLDS_GRID,
    RANDOM_SEED,
    MARGIN_THRESHOLD,
)
from src.features import extract_features_for_candidates, FEATURE_NAMES
from src.hard_negatives import sample_hard_negatives
from src.evaluate import evaluate_predictions
from src.utils import logger, time_block, evaluate_macro_f05


def create_labeled_dataset(
    candidates_dict: Dict[str, List[Tuple[str, float, int]]],
    ground_truth_dict: Dict[str, Set[str]],
    use_hard_negatives: bool = True
) -> Tuple[Dict[str, List[Tuple[str, float, int]]], np.ndarray, List[Tuple[str, str]]]:
    """
    Filter candidates into hard negative sampled training dataset.
    """
    filtered_cands = {}
    pair_ids = []
    labels = []

    for s1, cands in candidates_dict.items():
        true_set = ground_truth_dict.get(s1, set())
        if use_hard_negatives:
            pos_ids, neg_ids = sample_hard_negatives(s1, cands, true_set)
            sel_ids = set(pos_ids) | set(neg_ids)
            sel_cands = [c for c in cands if c[0] in sel_ids]
        else:
            sel_cands = cands

        filtered_cands[s1] = sel_cands
        for cand_id, score, support in sel_cands:
            pair_ids.append((s1, cand_id))
            labels.append(1 if cand_id in true_set else 0)

    y = np.array(labels, dtype=np.int32)
    positives = int(np.sum(y))
    negatives = len(y) - positives
    logger.info(f"Constructed labeled dataset: {len(y):,} pairs ({positives:,} positive, {negatives:,} negative | Pos ratio: {positives/max(1, len(y)):.2%})")

    return filtered_cands, y, pair_ids


@time_block("Singleton-Aware Threshold & Margin Optimization")
def find_optimal_threshold(
    val_s1_ids: List[str],
    val_pair_ids: List[Tuple[str, str]],
    val_probs: np.ndarray,
    ground_truth_dict: Dict[str, Set[str]],
    thresholds: List[float] = THRESHOLDS_GRID
) -> Tuple[float, float, float, Dict[float, float]]:
    """
    Grid search decision threshold and probability margin to maximize validation Macro F0.5.
    Penalizes false positives on singletons heavily.
    """
    logger.info("Optimizing threshold & margin for Macro F0.5...")

    val_s1_preds = defaultdict(list)
    for (s1, cand), prob in zip(val_pair_ids, val_probs):
        val_s1_preds[s1].append((cand, prob))

    for s1 in val_s1_preds:
        val_s1_preds[s1].sort(key=lambda x: x[1], reverse=True)

    best_thresh = 0.85
    best_margin = 0.05
    best_macro_f05 = -1.0
    thresh_scores = {}

    margin_candidates = [0.0, 0.02, 0.05, 0.08, 0.10]

    for thresh in thresholds:
        for margin in margin_candidates:
            pred_dict = {}
            for s1 in val_s1_ids:
                pairs = val_s1_preds.get(s1, [])
                matched = set()

                if pairs:
                    best_cand, best_prob = pairs[0]
                    if best_prob >= thresh:
                        second_prob = pairs[1][1] if len(pairs) > 1 else 0.0
                        if (best_prob - second_prob) >= margin or best_prob >= (thresh + 0.05):
                            matched.add(best_cand)

                            for cand, prob in pairs[1:]:
                                if prob >= thresh and (best_prob - prob) <= 0.03:
                                    matched.add(cand)

                pred_dict[s1] = matched

            eval_res = evaluate_macro_f05(
                ground_truth=ground_truth_dict,
                predictions=pred_dict,
                s1_ids=val_s1_ids
            )
            score = eval_res["macro_f05"]
            if margin == 0.05:
                thresh_scores[thresh] = score

            if score > best_macro_f05:
                best_macro_f05 = score
                best_thresh = thresh
                best_margin = margin

    logger.info(f"★ OPTIMAL THRESHOLD FOUND: threshold={best_thresh:.2f}, margin={best_margin:.2f} -> Validation Macro F0.5 = {best_macro_f05:.4f}")
    return best_thresh, best_margin, best_macro_f05, thresh_scores


@time_block("LightGBM Model Training Pipeline")
def train_matcher_model(
    candidates_dict: Dict[str, List[Tuple[str, float, int]]],
    s1_preprocessed: pd.DataFrame,
    targets_preprocessed: pd.DataFrame,
    ground_truth_dict: Dict[str, Set[str]],
    model_type: str = "lightgbm"
) -> Tuple[Any, float, float, Dict[str, Any]]:
    """
    Train LightGBM binary matcher, tune threshold/margin on validation split, and save model & metadata.
    """
    all_s1_ids = list(s1_preprocessed["entity_id"].unique())
    train_s1_ids, val_s1_ids = train_test_split(
        all_s1_ids,
        test_size=VAL_SPLIT_RATIO,
        random_state=RANDOM_SEED
    )
    logger.info(f"Entity-level Split: {len(train_s1_ids):,} Train S1 entities, {len(val_s1_ids):,} Validation S1 entities")

    train_s1_set = set(train_s1_ids)
    val_s1_set = set(val_s1_ids)

    raw_train_cands = {s1: cands for s1, cands in candidates_dict.items() if s1 in train_s1_set}
    raw_val_cands = {s1: cands for s1, cands in candidates_dict.items() if s1 in val_s1_set}

    train_cands, y_train, train_pair_ids = create_labeled_dataset(raw_train_cands, ground_truth_dict, use_hard_negatives=True)
    val_cands, y_val, val_pair_ids = create_labeled_dataset(raw_val_cands, ground_truth_dict, use_hard_negatives=False)

    logger.info("Extracting features for training pairs...")
    X_train, train_pair_ids = extract_features_for_candidates(train_cands, s1_preprocessed, targets_preprocessed)
    
    logger.info("Extracting features for validation pairs...")
    X_val, val_pair_ids = extract_features_for_candidates(val_cands, s1_preprocessed, targets_preprocessed)

    pos_count = int(np.sum(y_train))
    neg_count = len(y_train) - pos_count
    scale_pos = neg_count / max(1, pos_count)
    logger.info(f"Training LightGBM Classifier (scale_pos_weight={scale_pos:.2f})...")

    model = LGBMClassifier(
        n_estimators=LIGHTGBM_CONFIG["n_estimators"],
        learning_rate=LIGHTGBM_CONFIG["learning_rate"],
        num_leaves=LIGHTGBM_CONFIG["num_leaves"],
        max_depth=LIGHTGBM_CONFIG["max_depth"],
        min_child_samples=LIGHTGBM_CONFIG["min_child_samples"],
        subsample=LIGHTGBM_CONFIG["subsample"],
        colsample_bytree=LIGHTGBM_CONFIG["colsample_bytree"],
        reg_alpha=LIGHTGBM_CONFIG["reg_alpha"],
        reg_lambda=LIGHTGBM_CONFIG["reg_lambda"],
        scale_pos_weight=scale_pos,
        random_state=RANDOM_SEED,
        n_jobs=LIGHTGBM_CONFIG["n_jobs"],
        verbose=-1
    )
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1] if len(X_val) > 0 else np.zeros(0)

    best_thresh, best_margin, best_macro_f05, thresh_scores = find_optimal_threshold(
        val_s1_ids=val_s1_ids,
        val_pair_ids=val_pair_ids,
        val_probs=val_probs,
        ground_truth_dict=ground_truth_dict
    )

    val_pred_dict = {}
    val_s1_preds = defaultdict(list)
    for (s1, cand), prob in zip(val_pair_ids, val_probs):
        val_s1_preds[s1].append((cand, prob))

    for s1 in val_s1_ids:
        pairs = val_s1_preds.get(s1, [])
        matched = set()
        if pairs:
            pairs.sort(key=lambda x: x[1], reverse=True)
            best_cand, best_prob = pairs[0]
            if best_prob >= best_thresh:
                second_prob = pairs[1][1] if len(pairs) > 1 else 0.0
                if (best_prob - second_prob) >= best_margin or best_prob >= (best_thresh + 0.05):
                    matched.add(best_cand)
                    for cand, prob in pairs[1:]:
                        if prob >= best_thresh and (best_prob - prob) <= 0.03:
                            matched.add(cand)
        val_pred_dict[s1] = matched

    val_eval = evaluate_predictions(
        ground_truth=ground_truth_dict,
        predictions=val_pred_dict,
        s1_metadata_df=s1_preprocessed,
        s1_ids=val_s1_ids
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")

    metadata = {
        "model_type": "LightGBM",
        "feature_names": FEATURE_NAMES,
        "optimal_threshold": float(best_thresh),
        "optimal_margin": float(best_margin),
        "validation_macro_f05": float(best_macro_f05),
        "validation_metrics": val_eval,
        "threshold_scores": {str(k): float(v) for k, v in thresh_scores.items()},
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {METADATA_PATH}")

    return model, best_thresh, best_margin, metadata
