"""
Evaluation and diagnostic scoring module for Business Entity Resolution.
Implements the exact macro-averaged F0.5 metric per Source 1 entity.
"""

from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np
import pandas as pd
from src.utils import compute_entity_f05, evaluate_macro_f05, parse_id_list, logger, time_block


@time_block("Detailed Evaluation")
def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_metadata_df: Optional[pd.DataFrame] = None,
    s1_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation and print comprehensive diagnostic report.
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    eval_results = evaluate_macro_f05(
        ground_truth=ground_truth,
        predictions=predictions,
        s1_ids=s1_ids
    )

    # Compute global pair-level statistics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_pairs = 0
    total_pred_pairs = 0

    per_entity_stats = []

    for s1 in s1_ids:
        truth = ground_truth.get(s1, set())
        pred = predictions.get(s1, set())

        tp = len(truth & pred)
        fp = len(pred - truth)
        fn = len(truth - pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_true_pairs += len(truth)
        total_pred_pairs += len(pred)

        p, r, f05 = compute_entity_f05(truth, pred)
        per_entity_stats.append({
            "source1_entity_id": s1,
            "truth_count": len(truth),
            "pred_count": len(pred),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": p,
            "recall": r,
            "f05": f05,
            "is_singleton": len(truth) == 0,
        })

    pair_precision = (total_tp / total_pred_pairs) if total_pred_pairs > 0 else 0.0
    pair_recall = (total_tp / total_true_pairs) if total_true_pairs > 0 else 0.0
    denom = 0.25 * pair_precision + pair_recall
    pair_f05 = (1.25 * pair_precision * pair_recall) / denom if denom > 0 else 0.0

    eval_results.update({
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f05": pair_f05,
        "total_true_pairs": total_true_pairs,
        "total_predicted_pairs": total_pred_pairs,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    })

    # Country breakdown if metadata available
    country_metrics = {}
    if s1_metadata_df is not None and "country" in s1_metadata_df.columns:
        s1_country_map = dict(zip(s1_metadata_df["entity_id"], s1_metadata_df["country"]))
        for country in s1_metadata_df["country"].unique():
            cntry_s1 = [s for s in s1_ids if s1_country_map.get(s) == country]
            if cntry_s1:
                cntry_eval = evaluate_macro_f05(ground_truth, predictions, s1_ids=cntry_s1)
                country_metrics[country] = cntry_eval["macro_f05"]
        eval_results["country_macro_f05"] = country_metrics

    # Print summary
    logger.info("=" * 60)
    logger.info("OFFICIAL CHALLENGE EVALUATION RESULTS (PER S1 ENTITY MACRO)")
    logger.info("=" * 60)
    logger.info(f"  ★ MACRO F0.5 SCORE:      {eval_results['macro_f05']:.4f}")
    logger.info(f"  ★ Macro Precision:        {eval_results['macro_precision']:.4f}")
    logger.info(f"  ★ Macro Recall:           {eval_results['macro_recall']:.4f}")
    logger.info(f"  Singletons Accuracy:      {eval_results['singleton_accuracy']:.4f} ({eval_results['singletons_correct']:,} / {eval_results['singletons_total']:,})")
    logger.info(f"  Non-Singleton Entities:   {eval_results['non_singletons_total']:,}")
    logger.info(f"  Total Evaluated Entities: {eval_results['total_evaluated_s1']:,}")
    logger.info("-" * 60)
    logger.info(f"  Pair-level Precision:     {pair_precision:.4f} (TP: {total_tp:,}, FP: {total_fp:,})")
    logger.info(f"  Pair-level Recall:        {pair_recall:.4f} (TP: {total_tp:,}, FN: {total_fn:,})")
    logger.info(f"  Pair-level F0.5:          {pair_f05:.4f}")
    if country_metrics:
        logger.info("-" * 60)
        logger.info("  Breakdown by Country:")
        for cntry, score in country_metrics.items():
            logger.info(f"    - {cntry.upper()}: Macro F0.5 = {score:.4f}")
    logger.info("=" * 60)

    return eval_results