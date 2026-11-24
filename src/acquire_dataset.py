"""
Dataset acquisition script for FakeAVCeleb via HuggingFace.
Downloads, inspects, saves videos to disk, and creates subject-disjoint splits.
Adapted for thenewsupercell/fixed-fakeavceleb format.
"""
import os
import sys
import json
import hashlib
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_DIR, RAW_VIDEO_DIR, SPLITS_DIR, RANDOM_SEED

HF_DATASET = "thenewsupercell/fixed-fakeavceleb"
VIDEO_SAVE_DIR = os.path.join(RAW_VIDEO_DIR, "fakeavceleb")
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


def download_and_save_dataset(max_per_split=None):
    """
    Stream the HF dataset, save videos to disk, and build a metadata manifest.
    Uses streaming to avoid loading the entire dataset into memory.
    """
    from datasets import load_dataset
    import numpy as np
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"[Acquire] Loading {HF_DATASET} in streaming mode...")
    ds = load_dataset(HF_DATASET, streaming=True)
    print(f"[Acquire] Available splits: {list(ds.keys())}")

    manifest = {"train": [], "validation": [], "test": []}
    stats = Counter()

    for split_name in ds:
        print(f"\n[Acquire] Processing split: {split_name}")
        split_dir = os.path.join(VIDEO_SAVE_DIR, split_name)
        os.makedirs(split_dir, exist_ok=True)

        count = 0
        for i, sample in enumerate(ds[split_name]):
            if max_per_split and count >= max_per_split:
                break

            # Inspect first sample to understand structure
            if i == 0:
                print(f"  Sample keys: {list(sample.keys())}")
                for k, v in sample.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(f"    {k}: {v}")
                    elif isinstance(v, dict):
                        print(f"    {k}: dict with keys {list(v.keys())}")
                    elif isinstance(v, bytes):
                        print(f"    {k}: bytes, len={len(v)}")
                    elif hasattr(v, 'shape'):
                        print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"    {k}: type={type(v).__name__}")

            # Extract metadata
            entry = {}

            # Try to find label info
            for col in ["label", "target", "class", "is_fake", "fake",
                         "category", "type", "manipulation_type"]:
                if col in sample:
                    val = sample[col]
                    entry["raw_label"] = str(val)
                    if isinstance(val, int):
                        entry["label"] = "fake" if val == 1 else "real"
                        entry["label_int"] = val
                    elif isinstance(val, str):
                        entry["label"] = "fake" if "fake" in val.lower() else "real"
                    break

            # Subject / celebrity ID
            for col in ["celebrity", "subject", "identity", "person_id",
                         "subject_id", "source_celebrity"]:
                if col in sample:
                    entry["subject_id"] = str(sample[col])
                    break
            if "subject_id" not in entry:
                entry["subject_id"] = f"subj_{i % 50}"

            # Method / manipulation type
            for col in ["method", "manipulation", "race", "source",
                         "fake_type", "technique"]:
                if col in sample and col != "label":
                    entry["method"] = str(sample[col])
                    break
            if "method" not in entry:
                entry["method"] = "none"

            # Save video to disk
            video_data = None
            for col in ["video", "mp4", "file", "content"]:
                if col in sample:
                    val = sample[col]
                    if isinstance(val, dict):
                        # HF video format: {'path': ..., 'bytes': ...}
                        if 'bytes' in val:
                            video_data = val['bytes']
                            if 'path' in val:
                                entry["original_path"] = val['path']
                        elif 'path' in val:
                            entry["original_path"] = val['path']
                    elif isinstance(val, bytes):
                        video_data = val
                    break

            if video_data is None:
                # For torchcodec-decoded video, it comes as tensor frames
                # We'll need to re-encode or work with frames directly
                for col in ["video"]:
                    if col in sample:
                        val = sample[col]
                        if hasattr(val, '__len__') and not isinstance(val, (str, bytes)):
                            # It's decoded video frames
                            entry["is_decoded"] = True
                            entry["n_frames"] = len(val) if hasattr(val, '__len__') else 0
                            break

            # Create filename
            label_tag = entry.get("label", "unk")
            fname = f"{split_name}_{i:05d}_{label_tag}.mp4"
            fpath = os.path.join(split_dir, fname)
            entry["path"] = fpath
            entry["filename"] = fname
            entry["split"] = split_name
            entry["index"] = i

            if video_data is not None:
                with open(fpath, "wb") as f:
                    f.write(video_data)
                entry["saved"] = True
                entry["size_bytes"] = len(video_data)
            else:
                entry["saved"] = False

            manifest[split_name].append(entry)
            stats[entry.get("label", "unknown")] += 1
            count += 1

            if (count) % 100 == 0:
                print(f"  Processed {count} samples... ({dict(stats)})")

        print(f"  Split '{split_name}' done: {count} samples")

    # Print overall stats
    print(f"\n[Acquire] TOTAL STATS:")
    print(f"  Labels: {dict(stats)}")
    for split_name, entries in manifest.items():
        saved = sum(1 for e in entries if e.get("saved", False))
        print(f"  {split_name}: {len(entries)} total, {saved} videos saved to disk")

    return manifest


def create_splits_from_manifest(manifest):
    """
    Use the existing HF splits (train/val/test) since they already exist.
    Also verify subject disjointness.
    """
    import random
    random.seed(RANDOM_SEED)

    # Map HF split names to ours
    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }

    splits = {}
    for hf_name, our_name in split_map.items():
        if hf_name in manifest:
            splits[our_name] = manifest[hf_name]

    # Check subject disjointness
    split_subjects = {}
    for name, samples in splits.items():
        subjects = set(s.get("subject_id", "unk") for s in samples)
        split_subjects[name] = subjects
        print(f"  {name}: {len(samples)} samples, {len(subjects)} subjects")

    # Check overlap
    for n1 in split_subjects:
        for n2 in split_subjects:
            if n1 >= n2:
                continue
            overlap = split_subjects[n1] & split_subjects[n2]
            if overlap:
                print(f"  ⚠️  Subject overlap {n1}↔{n2}: {len(overlap)} subjects")
            else:
                print(f"  ✅ No subject overlap {n1}↔{n2}")

    # Save splits
    for name, data in splits.items():
        path = os.path.join(SPLITS_DIR, f"{name}.json")
        # Make JSON serializable
        clean_data = []
        for entry in data:
            clean = {k: v for k, v in entry.items()
                     if isinstance(v, (str, int, float, bool, type(None)))}
            clean_data.append(clean)
        with open(path, "w") as f:
            json.dump(clean_data, f, indent=2)
        print(f"  Saved {name}: {len(clean_data)} samples → {path}")

    return splits


def inspect_saved_videos(manifest):
    """Check how many videos were actually saved and are valid."""
    import cv2

    total = 0
    valid = 0
    invalid = 0

    for split_name, entries in manifest.items():
        for entry in entries[:10]:  # Check first 10 per split
            total += 1
            path = entry.get("path", "")
            if os.path.exists(path) and os.path.getsize(path) > 0:
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                    if frames > 0:
                        valid += 1
                        if total <= 3:
                            print(f"  ✅ {entry['filename']}: {frames:.0f} frames, "
                                  f"{w:.0f}x{h:.0f}, {fps:.1f} fps")
                    else:
                        invalid += 1
                else:
                    invalid += 1
                    cap.release()
            else:
                invalid += 1

    print(f"\n  Video check: {valid}/{total} valid (sampled)")
    return valid, total


if __name__ == "__main__":
    print("=" * 70)
    print("FakeAVCeleb Dataset Acquisition")
    print("=" * 70)

    # Start with a small batch to verify format, then go full
    print("\n[Phase 1] Downloading a small test batch (20 per split)...")
    manifest = download_and_save_dataset(max_per_split=20)

    print("\n[Phase 2] Inspecting saved videos...")
    inspect_saved_videos(manifest)

    print("\n[Phase 3] Creating splits...")
    splits = create_splits_from_manifest(manifest)

    print("\n✅ Test batch complete. Review the output above.")
    print("   If videos look good, re-run with max_per_split=None for full dataset.")
