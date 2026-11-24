"""
Training and Evaluation Pipeline for Multimodal Deepfake Detection
Evaluates all 7 ablation configurations, 3 fusion models, temporal aggregation,
per-manipulation breakdowns, and hard case error analysis.
"""
import os
import sys
import argparse
import json
import random
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, SCORE_CACHE_DIR, RESULTS_DIR, RANDOM_SEED
)

os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    auc = None
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) > 1:
                auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = None
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": auc,
        "confusion_matrix": cm
    }


class MLPFusionNet(nn.Module):
    def __init__(self, in_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_training_and_evaluation(features_path=None, test_size=0.3):
    if features_path is None:
        features_path = os.path.join(SCORE_CACHE_DIR, "extracted_features.pt")

    if not os.path.exists(features_path):
        print(f"[Error] Features file not found at: {features_path}")
        print("Please run `python src/extract_features.py` first.")
        return

    print("=" * 80)
    print("MULTIMODAL DEEPFAKE DETECTION — TRAINING & ABLATION EVALUATION")
    print("=" * 80)

    # 1. Load Extracted Features
    data = torch.load(features_path, map_location="cpu", weights_only=False)
    print(f"Loaded {len(data)} total samples from {features_path}")

    labels = np.array([d["label"] for d in data])
    cats = [d["category"] for d in data]
    print(f"Label balance: {Counter(labels.tolist())}")
    print(f"Categories: {Counter(cats)}")

    # 2. Stratified Train / Test Split
    indices = np.arange(len(data))
    stratify_labels = [f"{d['category']}_{d['label']}" for d in data]

    # Check if every class in stratify has at least 2 members
    strat_counts = Counter(stratify_labels)
    if min(strat_counts.values()) >= 2:
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=RANDOM_SEED, stratify=stratify_labels
        )
    else:
        train_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=RANDOM_SEED, stratify=labels
        )

    print(f"\nDataset Splits:")
    print(f"  Train Set: {len(train_indices)} samples ({dict(Counter(labels[train_indices].tolist()))})")
    print(f"  Test Set:  {len(test_indices)} samples ({dict(Counter(labels[test_indices].tolist()))})")

    # 3. Train Individual Branch Forensic Heads (A, V, E)
    print("\n[Step 1/5] Training Branch Forensic Classifiers (A, V, E)...")
    
    X_train_a = np.array([data[i]["a_feat"] for i in train_indices])
    X_train_v = np.array([data[i]["v_feat"] for i in train_indices])
    X_train_e = np.array([data[i]["e_feat"] for i in train_indices])
    y_train = labels[train_indices]

    X_test_a = np.array([data[i]["a_feat"] for i in test_indices])
    X_test_v = np.array([data[i]["v_feat"] for i in test_indices])
    X_test_e = np.array([data[i]["e_feat"] for i in test_indices])
    y_test = labels[test_indices]

    scaler_a = StandardScaler().fit(X_train_a)
    scaler_v = StandardScaler().fit(X_train_v)
    scaler_e = StandardScaler().fit(X_train_e)

    clf_a = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf_v = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)
    clf_e = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)

    clf_a.fit(scaler_a.transform(X_train_a), y_train)
    clf_v.fit(scaler_v.transform(X_train_v), y_train)
    clf_e.fit(scaler_e.transform(X_train_e), y_train)

    prob_train_a = clf_a.predict_proba(scaler_a.transform(X_train_a))[:, 1]
    prob_train_v = clf_v.predict_proba(scaler_v.transform(X_train_v))[:, 1]
    prob_train_e = clf_e.predict_proba(scaler_e.transform(X_train_e))[:, 1]

    prob_test_a = clf_a.predict_proba(scaler_a.transform(X_test_a))[:, 1]
    prob_test_v = clf_v.predict_proba(scaler_v.transform(X_test_v))[:, 1]
    prob_test_e = clf_e.predict_proba(scaler_e.transform(X_test_e))[:, 1]

    print("  Branch classifiers trained successfully.")

    # 4. Run All 7 Ablation Configurations
    print("\n[Step 2/5] Evaluating All 7 Ablation Configurations on Held-Out Test Set...")
    configs = {
        "A":     [("A", prob_train_a, prob_test_a)],
        "V":     [("V", prob_train_v, prob_test_v)],
        "E":     [("E", prob_train_e, prob_test_e)],
        "A+V":   [("A", prob_train_a, prob_test_a), ("V", prob_train_v, prob_test_v)],
        "A+E":   [("A", prob_train_a, prob_test_a), ("E", prob_train_e, prob_test_e)],
        "V+E":   [("V", prob_train_v, prob_test_v), ("E", prob_train_e, prob_test_e)],
        "A+V+E": [("A", prob_train_a, prob_test_a), ("V", prob_train_v, prob_test_v), ("E", prob_train_e, prob_test_e)],
    }

    ablation_results = {}

    for cfg_name, branch_list in configs.items():
        X_tr = np.column_stack([b[1] for b in branch_list])
        X_te = np.column_stack([b[2] for b in branch_list])

        cfg_res = {}

        # Fusion 1: Mean Averaging
        mean_te_prob = np.mean(X_te, axis=1)
        mean_te_pred = (mean_te_prob >= 0.5).astype(int)
        cfg_res["mean"] = compute_all_metrics(y_test, mean_te_pred, mean_te_prob)

        # Fusion 2: Regularized Logistic Regression
        lr_fuser = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=RANDOM_SEED)
        lr_fuser.fit(X_tr, y_train)
        lr_te_prob = lr_fuser.predict_proba(X_te)[:, 1]
        lr_te_pred = (lr_te_prob >= 0.5).astype(int)
        cfg_res["logistic"] = compute_all_metrics(y_test, lr_te_pred, lr_te_prob)
        cfg_res["logistic"]["weights"] = {b[0]: float(w) for b, w in zip(branch_list, lr_fuser.coef_[0])}

        # Fusion 3: Shallow MLP
        mlp = MLPFusionNet(in_features=X_tr.shape[1])
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-2, weight_decay=1e-4)
        criterion = nn.BCELoss()
        x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32)
        mlp.train()
        for epoch in range(80):
            optimizer.zero_grad()
            out = mlp(x_tr_t)
            loss = criterion(out, y_tr_t)
            loss.backward()
            optimizer.step()
        mlp.eval()
        with torch.no_grad():
            mlp_te_prob = mlp(torch.tensor(X_te, dtype=torch.float32)).numpy()
        mlp_te_pred = (mlp_te_prob >= 0.5).astype(int)
        cfg_res["mlp"] = compute_all_metrics(y_test, mlp_te_pred, mlp_te_prob)

        ablation_results[cfg_name] = cfg_res

    # 5. Temporal Aggregation Experiments (Mean vs Max vs Top-k on E windows)
    print("\n[Step 3/5] Evaluating Temporal Aggregation Strategies...")
    temporal_results = {}
    k_val = 2
    for agg_mode in ["mean", "max", f"top_{k_val}"]:
        te_agg_e = []
        for i in test_indices:
            divs = data[i]["window_divergences"][:, 1]
            if agg_mode == "mean":
                te_agg_e.append(np.mean(divs))
            elif agg_mode == "max":
                te_agg_e.append(np.max(divs))
            else:
                top_k = np.sort(divs)[-min(k_val, len(divs)):]
                te_agg_e.append(np.mean(top_k))

        te_agg_e = np.array(te_agg_e)
        te_agg_e_norm = (te_agg_e - te_agg_e.min()) / (te_agg_e.max() - te_agg_e.min() + 1e-6)
        agg_fused_prob = (prob_test_a + prob_test_v + te_agg_e_norm) / 3.0
        agg_pred = (agg_fused_prob >= 0.5).astype(int)
        temporal_results[agg_mode] = compute_all_metrics(y_test, agg_pred, agg_fused_prob)

    # 6. Manipulation-Type Stratified Analysis
    print("\n[Step 4/5] Computing Manipulation-Type Stratified Metrics...")
    test_categories = [data[i]["category"] for i in test_indices]
    stratified_results = {}
    fused_prob = (prob_test_a + prob_test_v + prob_test_e) / 3.0
    fused_pred = (fused_prob >= 0.5).astype(int)

    for cat_name in ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
        mask = np.array([c == cat_name for c in test_categories])
        if sum(mask) > 0:
            cat_y_true = y_test[mask]
            cat_y_pred = fused_pred[mask]
            cat_acc = float(np.mean(cat_y_true == cat_y_pred))
            rec = float(np.mean(cat_y_pred == 1)) if "Fake" in cat_name else float(np.mean(cat_y_pred == 0))
            stratified_results[cat_name] = {
                "count": int(sum(mask)),
                "accuracy": cat_acc,
                "recall": rec
            }
        else:
            stratified_results[cat_name] = {"count": 0, "accuracy": None, "recall": None}

    # 7. Hard-Case Error Analysis (>= 15 cases)
    print("\n[Step 5/5] Conducting Hard-Case Error Analysis (>= 15 clips)...")
    error_cases = []
    for rank_i, (test_i, p_a, p_v, p_e, p_tot, y_tr, y_pr) in enumerate(
        zip(test_indices, prob_test_a, prob_test_v, prob_test_e, fused_prob, y_test, fused_pred)
    ):
        vid = data[test_i]["video_id"]
        cat = data[test_i]["category"]
        is_err = (y_tr != y_pr)
        disagreement_score = float(np.std([p_a, p_v, p_e]))
        confidence_gap = float(abs(p_tot - 0.5))
        difficulty = (2.0 if is_err else 0.0) + disagreement_score * 1.5 - confidence_gap

        error_cases.append({
            "video_id": vid,
            "category": cat,
            "ground_truth": "Fake" if y_tr == 1 else "Real",
            "prediction": "Fake" if y_pr == 1 else "Real",
            "is_error": bool(is_err),
            "prob_audio": float(p_a),
            "prob_visual": float(p_v),
            "prob_emotion": float(p_e),
            "prob_fused": float(p_tot),
            "disagreement_std": disagreement_score,
            "difficulty": difficulty
        })

    error_cases = sorted(error_cases, key=lambda x: -x["difficulty"])
    top_15_cases = error_cases[:max(15, min(15, len(error_cases)))]

    final_output = {
        "ablation_results": ablation_results,
        "temporal_results": temporal_results,
        "stratified_results": stratified_results,
        "top_15_hard_cases": top_15_cases
    }

    out_json = os.path.join(RESULTS_DIR, "final_experimental_results.json")
    with open(out_json, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\n[Saved] All experimental results saved to {out_json}")

    print_results_table(ablation_results, temporal_results, stratified_results, top_15_cases)
    update_progress_report(ablation_results, temporal_results, stratified_results)


def print_results_table(ablations, temporal, stratified, hard_cases):
    print("\n" + "=" * 90)
    print("TABLE 1: 7 ABLATION CONFIGURATIONS COMPARISON (Logistic Fusion)")
    print("=" * 90)
    print(f"{'Config':<10} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ROC-AUC':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    for cfg in ["A", "V", "E", "A+V", "A+E", "V+E", "A+V+E"]:
        m = ablations[cfg]["logistic"]
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        print(f"{cfg:<10} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {auc_str:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")

    print("\n" + "=" * 90)
    print("TABLE 2: FUSION STRATEGY COMPARISON (Full Tripartite Model: A+V+E)")
    print("=" * 90)
    print(f"{'Fusion Method':<20} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    for fus in ["mean", "logistic", "mlp"]:
        m = ablations["A+V+E"][fus]
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        print(f"{fus.upper():<20} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {auc_str:<10}")

    print("\n" + "=" * 90)
    print("TABLE 3: TEMPORAL AGGREGATION COMPARISON")
    print("=" * 90)
    print(f"{'Aggregation Mode':<20} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ROC-AUC':<10}")
    print("-" * 60)
    for mode, m in temporal.items():
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        print(f"{mode.upper():<20} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {auc_str:<10}")

    print("\n" + "=" * 90)
    print("TABLE 4: MANIPULATION-TYPE STRATIFIED ACCURACY & RECALL")
    print("=" * 90)
    print(f"{'Manipulation Category':<25} | {'Count':<8} | {'Accuracy':<10} | {'Recall':<10}")
    print("-" * 65)
    for cat, m in stratified.items():
        acc_s = f"{m['accuracy']:.4f}" if m['accuracy'] is not None else "N/A"
        rec_s = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
        print(f"{cat:<25} | {m['count']:<8} | {acc_s:<10} | {rec_s:<10}")

    print("\n" + "=" * 90)
    print(f"TABLE 5: TOP {len(hard_cases)} HARD / AUDIT CASES (Evidence Analysis)")
    print("=" * 90)
    print(f"{'Video ID':<28} | {'Cat':<15} | {'True':<5} | {'Pred':<5} | {'P(A)':<6} | {'P(V)':<6} | {'P(E)':<6} | {'Err?':<5}")
    print("-" * 90)
    for c in hard_cases:
        err_str = "❌ YES" if c["is_error"] else "✅ NO"
        short_id = c["video_id"][:27]
        short_cat = c["category"].replace("Video", "V").replace("Audio", "A")[:14]
        print(f"{short_id:<28} | {short_cat:<15} | {c['ground_truth']:<5} | {c['prediction']:<5} | {c['prob_audio']:<6.3f} | {c['prob_visual']:<6.3f} | {c['prob_emotion']:<6.3f} | {err_str:<5}")


def update_progress_report(ablations, temporal, stratified):
    report_path = r"C:\Users\Satvick\.gemini\antigravity\brain\52c90c0a-4ede-492c-8e11-5d49d748032d\progress_report.md"
    if not os.path.exists(report_path):
        return

    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    rows = []
    for cfg in ["A", "V", "E", "A+V", "A+E", "V+E", "A+V+E"]:
        m = ablations[cfg]["logistic"]
        auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        rows.append(f"| {cfg:<6} | {m['accuracy']:.4f}   | {m['macro_f1']:.4f}   | {auc:<7} | {m['precision']:.4f}    | {m['recall']:.4f}  |")
    ablation_table = "\n".join(rows)

    t_rows = []
    for mode, m in temporal.items():
        auc = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "N/A"
        t_rows.append(f"| {mode:<11} | {m['accuracy']:.4f}   | {m['macro_f1']:.4f}   | {auc:<7} |")
    temp_table = "\n".join(t_rows)

    new_results_section = f"""
## Final Experimental Results

### Metrics across All 7 Ablation Configurations (Logistic Fusion)

| Config | Accuracy | Macro-F1 | ROC-AUC | Precision | Recall |
|--------|----------|----------|---------|-----------|--------|
{ablation_table}

### Temporal Aggregation Comparison (A+V+E Tripartite Model)

| Aggregation | Accuracy | Macro-F1 | ROC-AUC |
|-------------|----------|----------|---------|
{temp_table}

### Manipulation-Type Breakdown (Tripartite A+V+E)

| Manipulation Category | Count | Accuracy | Recall |
|-----------------------|-------|----------|--------|
| RealVideo-RealAudio | {stratified['RealVideo-RealAudio']['count']} | {stratified['RealVideo-RealAudio']['accuracy']} | {stratified['RealVideo-RealAudio']['recall']} |
| RealVideo-FakeAudio | {stratified['RealVideo-FakeAudio']['count']} | {stratified['RealVideo-FakeAudio']['accuracy']} | {stratified['RealVideo-FakeAudio']['recall']} |
| FakeVideo-RealAudio | {stratified['FakeVideo-RealAudio']['count']} | {stratified['FakeVideo-RealAudio']['accuracy']} | {stratified['FakeVideo-RealAudio']['recall']} |
| FakeVideo-FakeAudio | {stratified['FakeVideo-FakeAudio']['count']} | {stratified['FakeVideo-FakeAudio']['accuracy']} | {stratified['FakeVideo-FakeAudio']['recall']} |
"""
    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n" + new_results_section)
    print(f"[Report] Updated {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None, help="Path to cached features .pt file")
    parser.add_argument("--test_size", type=float, default=0.3, help="Fraction of dataset reserved for testing")
    args = parser.parse_args()

    run_training_and_evaluation(features_path=args.features, test_size=args.test_size)
