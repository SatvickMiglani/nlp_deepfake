"""
Main Pipeline — Multimodal Deepfake Detection
Orchestrates: data loading → feature extraction → score caching → fusion → evaluation
"""
import os
import sys
import json
import time
import hashlib
import traceback
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import (
    DEVICE, DATA_DIR, SCORE_CACHE_DIR, RESULTS_DIR, FIGURES_DIR,
    SPLITS_DIR, ABLATION_CONFIGS, RANDOM_SEED, SEGMENT_DURATION_SEC,
    SAMPLE_RATE, TEMPORAL_AGG_STRATEGIES
)

# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════════

def step1_load_dataset():
    """Load dataset and create splits."""
    print("\n" + "=" * 70)
    print("STEP 1: DATASET LOADING")
    print("=" * 70)

    from src.data_loader import (
        search_fakeavceleb_hf, try_load_fakeavceleb_hf,
        prepare_hf_dataset_for_pipeline, inspect_dataset,
        create_subject_disjoint_splits, save_splits, load_splits,
        load_from_local_directory
    )
    from configs.config import RAW_VIDEO_DIR

    # Check if splits already exist
    existing_splits = load_splits()
    if existing_splits and all(k in existing_splits for k in ["train", "val", "test"]):
        print("✅ Splits already exist, loading them.")
        return existing_splits

    # Search HuggingFace
    print("\nSearching HuggingFace for FakeAVCeleb...")
    hf_results = search_fakeavceleb_hf()
    for r in hf_results:
        print(f"  Found: {r['id']}")

    # Try loading from HuggingFace
    hf_ds = try_load_fakeavceleb_hf()
    if hf_ds is not None:
        samples = prepare_hf_dataset_for_pipeline(hf_ds)
    elif os.path.exists(RAW_VIDEO_DIR) and os.listdir(RAW_VIDEO_DIR):
        print(f"Loading from local directory: {RAW_VIDEO_DIR}")
        samples = load_from_local_directory(RAW_VIDEO_DIR)
    else:
        print("❌ No dataset found. Please provide FakeAVCeleb data.")
        return None

    stats = inspect_dataset(samples)
    splits = create_subject_disjoint_splits(samples)
    save_splits(splits)
    return splits


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Feature Extraction & Score Caching
# ═══════════════════════════════════════════════════════════════════════════════

def get_sample_id(sample):
    """Generate a unique sample ID."""
    path = sample.get("path", sample.get("filename", str(sample.get("hf_index", ""))))
    return hashlib.md5(path.encode()).hexdigest()[:12]


def step2_extract_audio_features(samples, cache_mgr):
    """Extract audio features for all samples."""
    print("\n" + "=" * 70)
    print("STEP 2a: AUDIO FEATURE EXTRACTION")
    print("=" * 70)

    from src.branches.audio_branch import AudioBranch

    branch = AudioBranch(device=DEVICE)
    processed = 0
    skipped = 0
    failed = 0

    for i, sample in enumerate(samples):
        sid = get_sample_id(sample)

        if cache_mgr.has_scores(sid, "audio"):
            skipped += 1
            continue

        path = sample.get("path", "")
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            scores = branch.get_segment_scores(path)
            if scores is not None:
                cache_mgr.save_scores(sid, "audio", scores)
                features = branch.get_features(path)
                if features is not None:
                    cache_mgr.save_features(sid, "audio", features)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(samples)} "
                  f"(processed={processed}, skipped={skipped}, failed={failed})")

    print(f"  ✅ Audio done: processed={processed}, skipped={skipped}, failed={failed}")
    return processed, skipped, failed


def step2_extract_visual_features(samples, cache_mgr):
    """Extract visual features for all samples."""
    print("\n" + "=" * 70)
    print("STEP 2b: VISUAL FEATURE EXTRACTION (MediaPipe)")
    print("=" * 70)

    from src.branches.visual_branch import VisualBranch

    branch = VisualBranch(device=DEVICE)
    processed = 0
    skipped = 0
    failed = 0

    for i, sample in enumerate(samples):
        sid = get_sample_id(sample)

        if cache_mgr.has_scores(sid, "visual"):
            skipped += 1
            continue

        path = sample.get("path", "")
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            scores = branch.get_segment_scores(path)
            if scores is not None:
                cache_mgr.save_scores(sid, "visual", scores)
                features = branch.get_features(path)
                if features is not None:
                    cache_mgr.save_features(sid, "visual", features)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(samples)} "
                  f"(processed={processed}, skipped={skipped}, failed={failed})")

    print(f"  ✅ Visual done: processed={processed}, skipped={skipped}, failed={failed}")
    return processed, skipped, failed


def step2_extract_emotion_features(samples, cache_mgr):
    """Extract emotion consistency features for all samples."""
    print("\n" + "=" * 70)
    print("STEP 2c: EMOTION CONSISTENCY EXTRACTION")
    print("=" * 70)

    from src.branches.emotion_branch import EmotionBranch

    branch = EmotionBranch(device=DEVICE)
    processed = 0
    skipped = 0
    failed = 0

    for i, sample in enumerate(samples):
        sid = get_sample_id(sample)

        if cache_mgr.has_scores(sid, "emotion"):
            skipped += 1
            continue

        path = sample.get("path", "")
        if not os.path.exists(path):
            failed += 1
            continue

        try:
            # Emotion branch needs both audio and video from the same file
            scores = branch.get_segment_scores(path, path)
            if scores is not None:
                cache_mgr.save_scores(sid, "emotion", scores)
                features = branch.get_features(path, path)
                if features is not None:
                    cache_mgr.save_features(sid, "emotion", features)
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            failed += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(samples)} "
                  f"(processed={processed}, skipped={skipped}, failed={failed})")

    print(f"  ✅ Emotion done: processed={processed}, skipped={skipped}, failed={failed}")
    return processed, skipped, failed


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Train Branch Classifiers
# ═══════════════════════════════════════════════════════════════════════════════

def step3_train_branch_classifiers(train_samples, val_samples, cache_mgr):
    """Train the classifier heads for each branch using cached features."""
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING BRANCH CLASSIFIERS")
    print("=" * 70)

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    results = {}

    for branch_name in ["audio", "visual", "emotion"]:
        print(f"\n--- Training {branch_name} classifier ---")

        # Collect features and labels
        X_train, y_train = [], []
        for sample in train_samples:
            sid = get_sample_id(sample)
            features = cache_mgr.load_features(sid, branch_name)
            if features is not None:
                label = 1 if sample["label"] == "fake" else 0
                # Use mean of segment features
                mean_feat = features.mean(axis=0) if features.ndim > 1 else features
                X_train.append(mean_feat)
                y_train.append(label)

        X_val, y_val = [], []
        for sample in val_samples:
            sid = get_sample_id(sample)
            features = cache_mgr.load_features(sid, branch_name)
            if features is not None:
                label = 1 if sample["label"] == "fake" else 0
                mean_feat = features.mean(axis=0) if features.ndim > 1 else features
                X_val.append(mean_feat)
                y_val.append(label)

        if not X_train:
            print(f"  ❌ No training features for {branch_name}")
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val) if X_val else None
        y_val = np.array(y_val) if y_val else None

        print(f"  Train: {X_train.shape}, Val: {X_val.shape if X_val is not None else 'None'}")

        # Train classifier
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=RANDOM_SEED)
        clf.fit(X_train, y_train)

        # Evaluate on validation
        if X_val is not None and len(X_val) > 0:
            val_preds = clf.predict(X_val)
            val_probs = clf.predict_proba(X_val)[:, 1]
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            acc = accuracy_score(y_val, val_preds)
            f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)
            try:
                auc = roc_auc_score(y_val, val_probs)
            except ValueError:
                auc = None
            print(f"  Val accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {auc}")
            results[branch_name] = {"accuracy": acc, "f1": f1, "auc": auc}

        # Re-score all samples using trained classifier
        print(f"  Re-scoring samples with trained {branch_name} classifier...")
        all_samples = train_samples + val_samples
        for sample in all_samples:
            sid = get_sample_id(sample)
            features = cache_mgr.load_features(sid, branch_name)
            if features is not None:
                if features.ndim == 1:
                    features = features.reshape(1, -1)
                scores = clf.predict_proba(features)[:, 1]
                cache_mgr.save_scores(sid, branch_name, scores)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Run Ablation Experiments
# ═══════════════════════════════════════════════════════════════════════════════

def step4_run_ablations(splits, cache_mgr):
    """Run all 7 ablation experiments."""
    print("\n" + "=" * 70)
    print("STEP 4: ABLATION EXPERIMENTS")
    print("=" * 70)

    from src.fusion.fusion_models import AblationEvaluator, print_comparison_table

    evaluator = AblationEvaluator(cache_mgr)

    train_ids = [get_sample_id(s) for s in splits["train"]]
    val_ids = [get_sample_id(s) for s in splits["val"]]
    test_ids = [get_sample_id(s) for s in splits["test"]]

    train_labels = [1 if s["label"] == "fake" else 0 for s in splits["train"]]
    val_labels = [1 if s["label"] == "fake" else 0 for s in splits["val"]]
    test_labels = [1 if s["label"] == "fake" else 0 for s in splits["test"]]

    # Run all configs with mean aggregation
    results = evaluator.run_all_configs(
        train_ids, train_labels, test_ids, test_labels,
        val_ids, val_labels, agg_strategy="mean"
    )

    # Print comparison table
    print_comparison_table(results, "logistic")

    # Temporal aggregation comparison
    print("\n\n--- Temporal Aggregation Comparison ---")
    temporal_results = evaluator.run_temporal_comparison(
        train_ids, train_labels, test_ids, test_labels,
        val_ids, val_labels
    )

    evaluator.results["temporal"] = temporal_results
    evaluator.save_results()

    return results, temporal_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Manipulation-Type Stratified Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def step5_stratified_analysis(splits, cache_mgr, ablation_results):
    """Analyze performance broken down by manipulation type."""
    print("\n" + "=" * 70)
    print("STEP 5: MANIPULATION-TYPE STRATIFIED ANALYSIS")
    print("=" * 70)

    from src.fusion.fusion_models import (
        AblationEvaluator, aggregate_segments,
        LogisticFusion, compute_metrics
    )

    test_samples = splits["test"]

    # Group test samples by manipulation type
    type_groups = {}
    for sample in test_samples:
        method = sample.get("method", "unknown")
        label = sample["label"]
        key = f"{label}_{method}" if method != "none" else label
        if key not in type_groups:
            type_groups[key] = []
        type_groups[key].append(sample)

    print(f"Manipulation types in test set:")
    for key, samples in type_groups.items():
        print(f"  {key}: {len(samples)} samples")

    stratified_results = {}
    for config_name in ["A", "V", "E", "A+V", "A+V+E"]:
        config = ABLATION_CONFIGS[config_name]
        active_branches = [b for b, active in config.items() if active]

        for type_key, type_samples in type_groups.items():
            if len(type_samples) < 3:
                continue

            scores_list = []
            labels = []
            for sample in type_samples:
                sid = get_sample_id(sample)
                row = []
                valid = True
                for branch in active_branches:
                    seg_scores = cache_mgr.load_scores(sid, branch)
                    if seg_scores is None:
                        valid = False
                        break
                    row.append(aggregate_segments(seg_scores, "mean"))
                if valid:
                    scores_list.append(row)
                    labels.append(1 if sample["label"] == "fake" else 0)

            if len(scores_list) >= 3 and len(set(labels)) > 1:
                X = np.array(scores_list)
                y = np.array(labels)
                preds = (np.mean(X, axis=1) >= 0.5).astype(int)
                metrics = compute_metrics(y, preds, np.mean(X, axis=1))
                key = f"{config_name}__{type_key}"
                stratified_results[key] = metrics
                print(f"  {config_name} on {type_key}: F1={metrics['macro_f1']:.4f}, "
                      f"Acc={metrics['accuracy']:.4f}")

    # Save
    path = os.path.join(RESULTS_DIR, "stratified_results.json")
    with open(path, "w") as f:
        json.dump(stratified_results, f, indent=2, default=str)
    print(f"✅ Stratified results saved to {path}")

    return stratified_results


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Hard Case Error Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def step6_error_analysis(splits, cache_mgr, n_cases=15):
    """Analyze at least 15 hard/error cases."""
    print("\n" + "=" * 70)
    print(f"STEP 6: ERROR ANALYSIS (top {n_cases} hard cases)")
    print("=" * 70)

    from src.fusion.fusion_models import aggregate_segments

    test_samples = splits["test"]
    error_cases = []

    for sample in test_samples:
        sid = get_sample_id(sample)
        true_label = 1 if sample["label"] == "fake" else 0

        # Get scores from all branches
        audio_scores = cache_mgr.load_scores(sid, "audio")
        visual_scores = cache_mgr.load_scores(sid, "visual")
        emotion_scores = cache_mgr.load_scores(sid, "emotion")

        a_score = aggregate_segments(audio_scores, "mean") if audio_scores is not None else None
        v_score = aggregate_segments(visual_scores, "mean") if visual_scores is not None else None
        e_score = aggregate_segments(emotion_scores, "mean") if emotion_scores is not None else None

        # Combined prediction
        valid_scores = [s for s in [a_score, v_score, e_score] if s is not None]
        if not valid_scores:
            continue

        combined_score = np.mean(valid_scores)
        combined_pred = 1 if combined_score >= 0.5 else 0

        # Compute disagreement among branches
        branch_preds = {}
        if a_score is not None:
            branch_preds["A"] = 1 if a_score >= 0.5 else 0
        if v_score is not None:
            branch_preds["V"] = 1 if v_score >= 0.5 else 0
        if e_score is not None:
            branch_preds["E"] = 1 if e_score >= 0.5 else 0

        n_agree = sum(1 for p in branch_preds.values() if p == combined_pred)
        n_disagree = len(branch_preds) - n_agree

        # Difficulty = prediction error + branch disagreement + confidence gap
        is_error = combined_pred != true_label
        confidence = abs(combined_score - 0.5)
        difficulty = (1.0 if is_error else 0.0) + n_disagree * 0.3 - confidence

        error_cases.append({
            "sample_id": sid,
            "path": sample.get("path", ""),
            "true_label": "fake" if true_label == 1 else "real",
            "predicted_label": "fake" if combined_pred == 1 else "real",
            "is_error": is_error,
            "combined_score": float(combined_score),
            "audio_score": float(a_score) if a_score is not None else None,
            "visual_score": float(v_score) if v_score is not None else None,
            "emotion_score": float(e_score) if e_score is not None else None,
            "branch_predictions": branch_preds,
            "n_disagreements": n_disagree,
            "difficulty": difficulty,
            "method": sample.get("method", "unknown"),
        })

    # Sort by difficulty (hardest first)
    error_cases.sort(key=lambda x: -x["difficulty"])

    print(f"\nTop {n_cases} hardest cases:")
    print("-" * 100)
    for i, case in enumerate(error_cases[:n_cases]):
        status = "❌ ERROR" if case["is_error"] else "✅ Correct"
        print(f"\n  Case {i+1}: {status}")
        print(f"    True: {case['true_label']}, Pred: {case['predicted_label']}")
        print(f"    Scores — A: {case['audio_score']}, V: {case['visual_score']}, E: {case['emotion_score']}")
        print(f"    Combined: {case['combined_score']:.4f}, Disagreements: {case['n_disagreements']}")
        print(f"    Method: {case['method']}")

    # Summary
    errors = [c for c in error_cases[:n_cases] if c["is_error"]]
    correct = [c for c in error_cases[:n_cases] if not c["is_error"]]
    branch_conflicts = [c for c in error_cases[:n_cases] if c["n_disagreements"] > 0]

    print(f"\n\nSummary of top {n_cases} cases:")
    print(f"  Errors: {len(errors)}")
    print(f"  Correct but hard: {len(correct)}")
    print(f"  With branch conflicts: {len(branch_conflicts)}")

    # Save
    path = os.path.join(RESULTS_DIR, "error_analysis.json")
    with open(path, "w") as f:
        json.dump(error_cases[:n_cases * 2], f, indent=2, default=str)
    print(f"\n✅ Error analysis saved to {path}")

    return error_cases[:n_cases]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run the complete multimodal deepfake detection pipeline."""
    start_time = time.time()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   MULTIMODAL DEEPFAKE DETECTION — FULL PIPELINE                ║")
    print("║   Audio + Visual + Emotion Consistency                         ║")
    print("║   7 Ablation Configurations                                    ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    from src.fusion.fusion_models import ScoreCacheManager
    cache_mgr = ScoreCacheManager()

    # Step 1: Dataset
    splits = step1_load_dataset()
    if splits is None:
        print("❌ Pipeline aborted: no dataset available.")
        return

    all_samples = splits["train"] + splits["val"] + splits["test"]

    # Step 2: Feature extraction
    step2_extract_audio_features(all_samples, cache_mgr)
    step2_extract_visual_features(all_samples, cache_mgr)
    step2_extract_emotion_features(all_samples, cache_mgr)

    # Step 3: Train branch classifiers
    branch_results = step3_train_branch_classifiers(splits["train"], splits["val"], cache_mgr)

    # Step 4: Ablation experiments
    ablation_results, temporal_results = step4_run_ablations(splits, cache_mgr)

    # Step 5: Stratified analysis
    stratified_results = step5_stratified_analysis(splits, cache_mgr, ablation_results)

    # Step 6: Error analysis
    error_cases = step6_error_analysis(splits, cache_mgr)

    elapsed = time.time() - start_time
    print(f"\n\n{'='*70}")
    print(f"PIPELINE COMPLETE — Total time: {elapsed/60:.1f} minutes")
    print(f"{'='*70}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Score cache: {SCORE_CACHE_DIR}")

    return {
        "branch_results": branch_results,
        "ablation_results": ablation_results,
        "temporal_results": temporal_results,
        "stratified_results": stratified_results,
        "error_cases": error_cases,
    }


if __name__ == "__main__":
    run_full_pipeline()
