"""
Dataset loader for FakeAVCeleb using HuggingFace API.
Handles downloading, inspection, and subject-disjoint splitting.
"""
import os
import sys
import json
import random
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DATA_DIR, SPLITS_DIR, DATASET_CACHE_DIR,
    VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, HF_DATASET_NAME
)


def try_load_fakeavceleb_hf():
    """
    Attempt to load FakeAVCeleb from HuggingFace.
    Returns a dataset dict or None if unavailable.
    """
    try:
        from datasets import load_dataset
        print(f"[DataLoader] Attempting to load '{HF_DATASET_NAME}' from HuggingFace...")
        ds = load_dataset(HF_DATASET_NAME, cache_dir=DATASET_CACHE_DIR, trust_remote_code=True)
        print(f"[DataLoader] Successfully loaded. Splits: {list(ds.keys())}")
        return ds
    except Exception as e:
        print(f"[DataLoader] HuggingFace load failed: {e}")
        return None


def search_fakeavceleb_hf():
    """
    Search HuggingFace Hub for FakeAVCeleb variants.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        results = api.list_datasets(search="FakeAVCeleb", limit=20)
        found = []
        for ds in results:
            found.append({
                "id": ds.id,
                "downloads": getattr(ds, "downloads", 0),
                "likes": getattr(ds, "likes", 0),
                "tags": getattr(ds, "tags", [])[:5],
            })
        return found
    except Exception as e:
        print(f"[DataLoader] HuggingFace search failed: {e}")
        return []


def load_from_local_directory(video_dir):
    """
    Load dataset from a local directory structure.
    Expected structure:
        video_dir/
          real/ or RealVideo/
            subject_id/
              clip.mp4
          fake/ or FakeVideo/
            method/
              subject_id/
                clip.mp4
    Returns a list of dicts with keys: path, label, subject_id, method
    """
    samples = []
    video_dir = Path(video_dir)

    if not video_dir.exists():
        print(f"[DataLoader] Directory {video_dir} does not exist.")
        return samples

    for root, dirs, files in os.walk(video_dir):
        for f in files:
            if not f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            fpath = os.path.join(root, f)
            rel = os.path.relpath(fpath, video_dir)
            parts = Path(rel).parts

            # Determine label from directory structure
            label = "unknown"
            subject_id = "unknown"
            method = "none"

            lower_parts = [p.lower() for p in parts]

            if any("real" in p for p in lower_parts):
                label = "real"
                # Try to extract subject ID
                for p in parts:
                    if p.startswith("id") or p.isdigit():
                        subject_id = p
                        break
            elif any("fake" in p for p in lower_parts):
                label = "fake"
                for p in parts:
                    if p.startswith("id") or p.isdigit():
                        subject_id = p
                        break
                # Try to determine manipulation method
                for p in lower_parts:
                    if "faceswap" in p or "fsgan" in p or "wav2lip" in p or "sv2tts" in p:
                        method = p
                        break

            samples.append({
                "path": fpath,
                "label": label,
                "subject_id": subject_id,
                "method": method,
                "filename": f,
            })

    print(f"[DataLoader] Found {len(samples)} video files in {video_dir}")
    return samples


def create_subject_disjoint_splits(samples, val_ratio=VAL_SPLIT, test_ratio=TEST_SPLIT, seed=RANDOM_SEED):
    """
    Create subject-disjoint train/val/test splits.
    Ensures no subject appears in more than one split.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Group by subject
    subject_samples = defaultdict(list)
    for s in samples:
        subject_samples[s["subject_id"]].append(s)

    subjects = list(subject_samples.keys())
    random.shuffle(subjects)

    n = len(subjects)
    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))

    test_subjects = set(subjects[:n_test])
    val_subjects = set(subjects[n_test:n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    splits = {"train": [], "val": [], "test": []}
    for s in samples:
        sid = s["subject_id"]
        if sid in test_subjects:
            splits["test"].append(s)
        elif sid in val_subjects:
            splits["val"].append(s)
        else:
            splits["train"].append(s)

    print(f"[DataLoader] Split sizes — Train: {len(splits['train'])}, "
          f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    print(f"[DataLoader] Subject counts — Train: {len(train_subjects)}, "
          f"Val: {len(val_subjects)}, Test: {len(test_subjects)}")

    # Check label distribution per split
    for split_name, split_data in splits.items():
        labels = Counter(s["label"] for s in split_data)
        print(f"  {split_name}: {dict(labels)}")

    return splits


def save_splits(splits, output_dir=SPLITS_DIR):
    """Save splits as JSON files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[DataLoader] Saved {name} split ({len(data)} samples) to {path}")


def load_splits(splits_dir=SPLITS_DIR):
    """Load previously saved splits."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(splits_dir, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                splits[name] = json.load(f)
            print(f"[DataLoader] Loaded {name}: {len(splits[name])} samples")
    return splits


def inspect_dataset(samples):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"\nTotal samples: {len(samples)}")

    labels = Counter(s["label"] for s in samples)
    print(f"\nLabel distribution:")
    for label, count in sorted(labels.items()):
        print(f"  {label}: {count} ({100*count/len(samples):.1f}%)")

    subjects = set(s["subject_id"] for s in samples)
    print(f"\nUnique subjects: {len(subjects)}")

    methods = Counter(s.get("method", "none") for s in samples)
    print(f"\nManipulation methods:")
    for method, count in sorted(methods.items()):
        print(f"  {method}: {count}")

    print("=" * 60 + "\n")
    return {
        "total": len(samples),
        "labels": dict(labels),
        "subjects": len(subjects),
        "methods": dict(methods),
    }


def prepare_hf_dataset_for_pipeline(hf_dataset):
    """
    Convert a HuggingFace dataset to our internal sample format.
    Adapts to whatever columns/structure the HF dataset provides.
    """
    samples = []
    # Inspect columns
    split_name = list(hf_dataset.keys())[0]
    columns = hf_dataset[split_name].column_names
    print(f"[DataLoader] HF dataset columns: {columns}")

    for split_name in hf_dataset:
        ds_split = hf_dataset[split_name]
        for i, row in enumerate(ds_split):
            sample = {
                "hf_index": i,
                "hf_split": split_name,
            }
            # Try to find video path/data
            for col in ["video", "video_path", "file", "path", "filename"]:
                if col in row:
                    sample["path"] = row[col] if isinstance(row[col], str) else f"hf_{split_name}_{i}"
                    break

            # Try to find label
            for col in ["label", "class", "target", "fake", "is_fake"]:
                if col in row:
                    val = row[col]
                    if isinstance(val, int):
                        sample["label"] = "fake" if val == 1 else "real"
                    elif isinstance(val, str):
                        sample["label"] = "fake" if "fake" in val.lower() else "real"
                    else:
                        sample["label"] = str(val)
                    break

            # Try to find subject
            for col in ["subject", "subject_id", "identity", "person_id", "celebrity"]:
                if col in row:
                    sample["subject_id"] = str(row[col])
                    break
            if "subject_id" not in sample:
                sample["subject_id"] = f"subj_{i % 50}"

            # Try to find method
            for col in ["method", "manipulation", "type", "category"]:
                if col in row:
                    sample["method"] = str(row[col])
                    break
            if "method" not in sample:
                sample["method"] = "none"

            samples.append(sample)

    return samples


if __name__ == "__main__":
    print("=" * 60)
    print("FakeAVCeleb Dataset Loader")
    print("=" * 60)

    # Step 1: Search HuggingFace for available versions
    print("\n[Step 1] Searching HuggingFace for FakeAVCeleb datasets...")
    hf_results = search_fakeavceleb_hf()
    if hf_results:
        print(f"Found {len(hf_results)} datasets on HuggingFace:")
        for r in hf_results:
            print(f"  - {r['id']} (downloads: {r['downloads']}, likes: {r['likes']})")
    else:
        print("No FakeAVCeleb datasets found on HuggingFace.")

    # Step 2: Try to load the primary dataset
    print("\n[Step 2] Attempting to load primary dataset...")
    hf_ds = try_load_fakeavceleb_hf()

    if hf_ds is not None:
        samples = prepare_hf_dataset_for_pipeline(hf_ds)
        stats = inspect_dataset(samples)
        splits = create_subject_disjoint_splits(samples)
        save_splits(splits)
        print("\n✅ Dataset loaded from HuggingFace and splits created!")
    else:
        # Fallback: check for local data
        from configs.config import RAW_VIDEO_DIR
        if os.path.exists(RAW_VIDEO_DIR) and os.listdir(RAW_VIDEO_DIR):
            print(f"\n[Fallback] Loading from local directory: {RAW_VIDEO_DIR}")
            samples = load_from_local_directory(RAW_VIDEO_DIR)
            if samples:
                stats = inspect_dataset(samples)
                splits = create_subject_disjoint_splits(samples)
                save_splits(splits)
                print("\n✅ Dataset loaded from local directory and splits created!")
            else:
                print("❌ No valid video files found in local directory.")
        else:
            print("\n❌ No dataset available. Options:")
            print("   1. Check HuggingFace access permissions for FakeAVCeleb")
            print("   2. Download manually and place videos in:", RAW_VIDEO_DIR)
            print("   3. Use a different HuggingFace dataset (update config.py)")
