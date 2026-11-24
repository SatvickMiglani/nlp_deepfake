import json

with open("outputs/results/final_experimental_results.json") as f:
    res = json.load(f)

print("Keys in result JSON:", list(res.keys()))

print("\n" + "="*85)
print("TABLE 1: 7 ABLATION CONFIGURATIONS (Logistic Fusion)")
print("="*85)
print(f"{'Config':<8} | {'Accuracy':<10} | {'Macro-F1':<10} | {'ROC-AUC':<10} | {'Precision':<10} | {'Recall':<10}")
print("-" * 75)
for cfg, methods in res["ablation_results"].items():
    m = methods["logistic"]
    auc_val = m.get("roc_auc")
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
    print(f"{cfg:<8} | {m['accuracy']:<10.4f} | {m['macro_f1']:<10.4f} | {auc_str:<10} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")

print("\n" + "="*85)
print("TABLE 2: FUSION STRATEGY COMPARISON (Full A+V+E)")
print("="*85)
for method, m in res["ablation_results"]["A+V+E"].items():
    auc_val = m.get("roc_auc")
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
    print(f"{method.upper():<10} | Acc: {m['accuracy']:<10.4f} | F1: {m['macro_f1']:<10.4f} | AUC: {auc_str:<10}")

print("\n" + "="*85)
print("TABLE 3: TEMPORAL AGGREGATION COMPARISON")
print("="*85)
for mode, m in res["temporal_results"].items():
    auc_val = m.get("roc_auc")
    auc_str = f"{auc_val:.4f}" if auc_val is not None else "N/A"
    print(f"{mode.upper():<10} | Acc: {m['accuracy']:<10.4f} | F1: {m['macro_f1']:<10.4f} | AUC: {auc_str:<10}")

print("\n" + "="*85)
print("TABLE 4: MANIPULATION-TYPE STRATIFIED RESULTS")
print("="*85)
for cat, m in res["stratified_results"].items():
    acc_s = f"{m['accuracy']:.4f}" if m['accuracy'] is not None else "N/A"
    rec_s = f"{m['recall']:.4f}" if m['recall'] is not None else "N/A"
    print(f"{cat:<25} | Count: {m['count']:<4} | Acc: {acc_s:<10} | Recall: {rec_s:<10}")

print("\n" + "="*85)
print("TABLE 5: TOP HARD / AUDIT CASES")
print("="*85)
hard_cases = res.get("top_15_hard_cases", [])
print(f"Total hard cases saved: {len(hard_cases)}")
for i, c in enumerate(hard_cases):
    err = "❌ YES" if c["is_error"] else "✅ NO"
    print(f"[{i+1:02d}] {c['video_id'][:28]:<28} | Cat: {c['category'][:15]:<15} | True: {c['ground_truth']:<4} | Pred: {c['prediction']:<4} | P(A): {c['prob_audio']:.3f} | P(V): {c['prob_visual']:.3f} | P(E): {c['prob_emotion']:.3f} | Err: {err}")

# Also check how many samples are in extracted_features.pt
import torch
feat_path = "outputs/scores/extracted_features.pt"
if torch.os.path.exists(feat_path):
    data = torch.load(feat_path, map_location="cpu", weights_only=False)
    print(f"\nCurrent extracted_features.pt contains: {len(data)} videos")
    from collections import Counter
    print("Categories in extracted_features.pt:", dict(Counter(d['category'] for d in data)))
