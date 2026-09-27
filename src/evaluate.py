"""
Evaluation and report generation engine for Business Entity Resolution.
Computes official Macro F0.5 per S1 entity and exports validation reports & error analysis CSV.
"""

from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
import numpy as np
import pandas as pd

from src.config import VALIDATION_REPORT_PATH, ERROR_ANALYSIS_PATH
from src.utils import compute_entity_f05, evaluate_macro_f05, parse_id_list, logger, time_block


@time_block("Detailed Evaluation & Report Export")
def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    s1_metadata_df: Optional[pd.DataFrame] = None,
    s1_ids: Optional[List[str]] = None,
    candidates_dict: Optional[Dict[str, List[Tuple[str, float, int]]]] = None,
    candidate_probs: Optional[Dict[Tuple[str, str], float]] = None,
) -> Dict[str, Any]:
    """
    Run full evaluation, print comprehensive diagnostic report, and write:
    - reports/validation_report.txt
    - reports/error_analysis.csv
    """
    if s1_ids is None:
        s1_ids = list(ground_truth.keys())

    eval_results = evaluate_macro_f05(
        ground_truth=ground_truth,
        predictions=predictions,
        s1_ids=s1_ids
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_true_pairs = 0
    total_pred_pairs = 0

    per_entity_rows = []

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
        per_entity_rows.append({
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

    # Country breakdown
    country_metrics = {}
    s1_meta_map = {}
    if s1_metadata_df is not None and "country" in s1_metadata_df.columns:
        s1_meta_map = {row.entity_id: row._asdict() for row in s1_metadata_df.itertuples(index=False)}
        for country in s1_metadata_df["country"].unique():
            cntry_s1 = [s for s in s1_ids if s1_meta_map.get(s, {}).get("country") == country]
            if cntry_s1:
                cntry_eval = evaluate_macro_f05(ground_truth, predictions, s1_ids=cntry_s1)
                country_metrics[country] = cntry_eval["macro_f05"]
        eval_results["country_macro_f05"] = country_metrics

    # Print summary
    logger.info("=" * 60)
    logger.info("OFFICIAL CHALLENGE EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  ★ MACRO F0.5 SCORE:      {eval_results['macro_f05']:.4f}")
    logger.info(f"  ★ Macro Precision:        {eval_results['macro_precision']:.4f}")
    logger.info(f"  ★ Macro Recall:           {eval_results['macro_recall']:.4f}")
    logger.info(f"  Singletons Accuracy:      {eval_results['singleton_accuracy']:.4f} ({eval_results['singletons_correct']:,} / {eval_results['singletons_total']:,})")
    logger.info(f"  Non-Singleton Entities:   {eval_results['non_singletons_total']:,}")
    logger.info(f"  Total Evaluated S1:       {eval_results['total_evaluated_s1']:,}")
    if country_metrics:
        logger.info("  Country Breakdown:")
        for cntry, score in country_metrics.items():
            logger.info(f"    - {cntry.upper()}: Macro F0.5 = {score:.4f}")
    logger.info("=" * 60)

    # Write reports/validation_report.txt
    VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(VALIDATION_REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write("OFFICIAL VALIDATION PERFORMANCE REPORT\n")
        f.write("==================================================\n\n")
        f.write(f"Macro F0.5 Score:             {eval_results['macro_f05']:.4f}\n")
        f.write(f"Macro Precision:              {eval_results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall:                 {eval_results['macro_recall']:.4f}\n")
        f.write(f"Singleton Accuracy:           {eval_results['singleton_accuracy']:.4f} ({eval_results['singletons_correct']}/{eval_results['singletons_total']})\n")
        f.write(f"Non-Singleton S1 Entities:    {eval_results['non_singletons_total']:,}\n")
        f.write(f"Total Evaluated Entities:     {eval_results['total_evaluated_s1']:,}\n\n")
        f.write(f"Pair-Level Precision:         {pair_precision:.4f}\n")
        f.write(f"Pair-Level Recall:            {pair_recall:.4f}\n")
        f.write(f"Pair-Level F0.5:              {pair_f05:.4f}\n\n")
        if country_metrics:
            f.write("Country Breakdown:\n")
            for cntry, score in country_metrics.items():
                f.write(f"  - {cntry.upper()}: Macro F0.5 = {score:.4f}\n")
    logger.info(f"Saved validation report to {VALIDATION_REPORT_PATH}")

    # Write reports/error_analysis.csv (worst 100 validation entities)
    error_entities = [e for e in per_entity_rows if e["f05"] < 1.0]
    error_entities.sort(key=lambda x: x["f05"])

    error_analysis_rows = []
    for entry in error_entities[:200]:
        s1 = entry["source1_entity_id"]
        meta = s1_meta_map.get(s1, {})
        truth_set = ground_truth.get(s1, set())
        pred_set = predictions.get(s1, set())

        fp_set = pred_set - truth_set
        fn_set = truth_set - pred_set

        error_analysis_rows.append({
            "source1_entity_id": s1,
            "f05": entry["f05"],
            "precision": entry["precision"],
            "recall": entry["recall"],
            "name_raw": meta.get("name_raw", ""),
            "address_norm": meta.get("address_norm", ""),
            "country": meta.get("country", ""),
            "true_matches": ",".join(sorted(list(truth_set))),
            "predicted_matches": ",".join(sorted(list(pred_set))),
            "false_positives": ",".join(sorted(list(fp_set))),
            "false_negatives": ",".join(sorted(list(fn_set))),
        })

    ERROR_ANALYSIS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(error_analysis_rows).to_csv(ERROR_ANALYSIS_PATH, index=False, encoding="utf-8")
    logger.info(f"Saved error analysis to {ERROR_ANALYSIS_PATH}")

    return eval_results