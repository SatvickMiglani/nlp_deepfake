"""
Results Compiler — Generate final comparison tables and summary report.
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import RESULTS_DIR, FIGURES_DIR, ABLATION_CONFIGS


def compile_results():
    """Load all results and produce a comprehensive comparison."""

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          FINAL RESULTS COMPILATION                             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # Load ablation results
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    if not os.path.exists(ablation_path):
        print("❌ No ablation results found. Run the pipeline first.")
        return None

    with open(ablation_path) as f:
        results = json.load(f)

    # ─── Table 1: Main Ablation Comparison ────────────────────────────────
    print("\n" + "=" * 90)
    print("TABLE 1: ABLATION COMPARISON — ALL 7 CONFIGURATIONS")
    print("=" * 90)

    for fusion_method in ["mean", "logistic", "mlp"]:
        print(f"\n--- Fusion: {fusion_method.upper()} ---")
        print(f"{'Config':<10} {'Accuracy':<10} {'Macro-F1':<10} {'ROC-AUC':<10} "
              f"{'Precision':<11} {'Recall':<10}")
        print("-" * 70)

        for config_name in ["A", "V", "E", "A+V", "A+E", "V+E", "A+V+E"]:
            if config_name in results and fusion_method in results[config_name]:
                m = results[config_name][fusion_method]
                auc = f"{m.get('roc_auc', 'N/A')}" if m.get('roc_auc') else "N/A"
                if isinstance(auc, float):
                    auc = f"{auc:.4f}"
                print(f"{config_name:<10} {m['accuracy']:<10.4f} {m['macro_f1']:<10.4f} "
                      f"{auc:<10} {m['precision']:<11.4f} {m['recall']:<10.4f}")

    # ─── Table 2: RQ1 — Audio+Visual vs Unimodal ─────────────────────────
    print("\n\n" + "=" * 90)
    print("TABLE 2: RQ1 — DOES A+V OUTPERFORM UNIMODAL?")
    print("=" * 90)

    for fusion_method in ["logistic"]:
        if all(c in results for c in ["A", "V", "A+V"]):
            a = results["A"].get(fusion_method, {})
            v = results["V"].get(fusion_method, {})
            av = results["A+V"].get(fusion_method, {})

            print(f"\n  A-only:  F1={a.get('macro_f1', 'N/A'):.4f}  AUC={a.get('roc_auc', 'N/A')}")
            print(f"  V-only:  F1={v.get('macro_f1', 'N/A'):.4f}  AUC={v.get('roc_auc', 'N/A')}")
            print(f"  A+V:     F1={av.get('macro_f1', 'N/A'):.4f}  AUC={av.get('roc_auc', 'N/A')}")

            av_f1 = av.get('macro_f1', 0)
            max_uni = max(a.get('macro_f1', 0), v.get('macro_f1', 0))
            delta = av_f1 - max_uni
            print(f"\n  → A+V vs best unimodal: Δ F1 = {delta:+.4f}")
            print(f"  → {'YES' if delta > 0 else 'NO'}: A+V {'outperforms' if delta > 0 else 'does not outperform'} unimodal baselines")

    # ─── Table 3: RQ2 — Does E help A+V? ─────────────────────────────────
    print("\n\n" + "=" * 90)
    print("TABLE 3: RQ2 — DOES EMOTION CONSISTENCY HELP?")
    print("=" * 90)

    for fusion_method in ["logistic"]:
        if all(c in results for c in ["A+V", "A+V+E"]):
            av = results["A+V"].get(fusion_method, {})
            ave = results["A+V+E"].get(fusion_method, {})

            print(f"\n  A+V:     F1={av.get('macro_f1', 'N/A'):.4f}  AUC={av.get('roc_auc', 'N/A')}")
            print(f"  A+V+E:   F1={ave.get('macro_f1', 'N/A'):.4f}  AUC={ave.get('roc_auc', 'N/A')}")

            delta_f1 = ave.get('macro_f1', 0) - av.get('macro_f1', 0)
            print(f"\n  → Emotion increment: Δ F1 = {delta_f1:+.4f}")
            if delta_f1 > 0.01:
                print(f"  → YES: Emotion consistency provides measurable improvement")
            elif delta_f1 > -0.01:
                print(f"  → MARGINAL: Emotion consistency has negligible effect")
            else:
                print(f"  → NO: Emotion consistency hurts performance")

    # ─── Table 4: Temporal Aggregation ────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("TABLE 4: TEMPORAL AGGREGATION COMPARISON (A+V+E)")
    print("=" * 90)

    temporal = results.get("temporal", {})
    if temporal:
        print(f"\n{'Strategy':<12} {'Accuracy':<10} {'Macro-F1':<10} {'ROC-AUC':<10}")
        print("-" * 50)
        for strategy in ["mean", "max", "top_k"]:
            if strategy in temporal:
                m = temporal[strategy].get("logistic", {})
                auc = m.get('roc_auc', 'N/A')
                if isinstance(auc, float):
                    auc = f"{auc:.4f}"
                print(f"{strategy:<12} {m.get('accuracy', 0):<10.4f} "
                      f"{m.get('macro_f1', 0):<10.4f} {auc:<10}")

    # ─── Table 5: Manipulation-Type Breakdown ─────────────────────────────
    print("\n\n" + "=" * 90)
    print("TABLE 5: PERFORMANCE BY MANIPULATION TYPE")
    print("=" * 90)

    strat_path = os.path.join(RESULTS_DIR, "stratified_results.json")
    if os.path.exists(strat_path):
        with open(strat_path) as f:
            strat = json.load(f)
        print(f"\n{'Config__Type':<35} {'Accuracy':<10} {'Macro-F1':<10}")
        print("-" * 60)
        for key in sorted(strat.keys()):
            m = strat[key]
            print(f"{key:<35} {m.get('accuracy', 0):<10.4f} {m.get('macro_f1', 0):<10.4f}")

    # ─── Error Analysis Summary ───────────────────────────────────────────
    print("\n\n" + "=" * 90)
    print("ERROR ANALYSIS SUMMARY")
    print("=" * 90)

    err_path = os.path.join(RESULTS_DIR, "error_analysis.json")
    if os.path.exists(err_path):
        with open(err_path) as f:
            errors = json.load(f)[:15]
        n_errors = sum(1 for e in errors if e.get("is_error"))
        n_conflicts = sum(1 for e in errors if e.get("n_disagreements", 0) > 0)
        print(f"\n  Analyzed: {len(errors)} hard cases")
        print(f"  Misclassified: {n_errors}")
        print(f"  With branch conflicts: {n_conflicts}")

        # Pattern analysis
        error_methods = [e["method"] for e in errors if e.get("is_error")]
        if error_methods:
            from collections import Counter
            method_counts = Counter(error_methods)
            print(f"  Error distribution by method: {dict(method_counts)}")

    # Save compiled report
    report_path = os.path.join(RESULTS_DIR, "final_report.txt")
    print(f"\n\n✅ Full report would be saved to: {report_path}")

    return results


if __name__ == "__main__":
    compile_results()
