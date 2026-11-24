"""
Dataset acquisition — bypasses torchcodec by reading parquet files directly.
Downloads raw video bytes from HuggingFace parquet files.
"""
import os
import sys
import json
import io
import random
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import DATA_DIR, RAW_VIDEO_DIR, SPLITS_DIR, RANDOM_SEED

HF_DATASET = "thenewsupercell/fixed-fakeavceleb"
VIDEO_SAVE_DIR = os.path.join(RAW_VIDEO_DIR, "fakeavceleb")
os.makedirs(VIDEO_SAVE_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)


def get_parquet_urls():
    """Get the parquet file URLs for each split from the HF Hub."""
    from huggingface_hub import HfApi
    api = HfApi()

    # List all files in the dataset repo
    files = api.list_repo_tree(HF_DATASET, repo_type="dataset")
    parquet_files = {}

    for f in files:
        name = f.path if hasattr(f, 'path') else str(f)
        if name.endswith('.parquet'):
            # Determine split from path
            for split in ['train', 'validation', 'test']:
                if split in name.lower():
                    if split not in parquet_files:
                        parquet_files[split] = []
                    parquet_files[split].append(name)
                    break

    print(f"[Acquire] Found parquet files: { {k: len(v) for k, v in parquet_files.items()} }")
    return parquet_files


def download_and_process_split(split_name, parquet_paths, max_samples=None):
    """
    Download a split's parquet files and extract video bytes + metadata.
    Skips torchcodec entirely by reading raw bytes.
    """
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    split_dir = os.path.join(VIDEO_SAVE_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    samples = []
    total = 0

    for pq_path in parquet_paths:
        print(f"  Downloading: {pq_path}...")
        local_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=pq_path,
            repo_type="dataset",
        )

        # Read parquet without datasets library
        table = pq.read_table(local_path)
        columns = table.column_names
        print(f"  Columns: {columns}")

        df = table.to_pandas()
        print(f"  Rows: {len(df)}")

        if total == 0:
            # Print first row info
            for col in columns:
                val = df[col].iloc[0]
                if isinstance(val, (str, int, float, bool, np.integer, np.floating)):
                    print(f"    {col}: {val}")
                elif isinstance(val, bytes):
                    print(f"    {col}: bytes, len={len(val)}")
                elif isinstance(val, dict):
                    print(f"    {col}: dict, keys={list(val.keys())}")
                else:
                    print(f"    {col}: type={type(val).__name__}")

        for idx, row in df.iterrows():
            if max_samples and total >= max_samples:
                break

            entry = {"split": split_name, "index": total}

            # Extract label
            for col in ["label", "target", "class", "is_fake", "fake", "type"]:
                if col in columns:
                    val = row[col]
                    entry["raw_label"] = str(val)
                    if isinstance(val, (int, np.integer)):
                        entry["label"] = "fake" if val == 1 else "real"
                        entry["label_int"] = int(val)
                    elif isinstance(val, str):
                        if "real" in val.lower() or val == "0":
                            entry["label"] = "real"
                        else:
                            entry["label"] = "fake"
                    break

            # Subject ID
            for col in ["celebrity", "subject", "identity", "person_id", "source_celebrity"]:
                if col in columns:
                    entry["subject_id"] = str(row[col])
                    break
            if "subject_id" not in entry:
                entry["subject_id"] = f"subj_{total % 50}"

            # Method
            for col in ["method", "manipulation", "technique", "fake_type"]:
                if col in columns and col not in ["label"]:
                    val = row[col]
                    if val is not None and str(val) != 'nan':
                        entry["method"] = str(val)
                        break
            if "method" not in entry:
                entry["method"] = "none"

            # Extract and save video bytes
            video_saved = False
            for col in ["video", "mp4", "file", "content"]:
                if col not in columns:
                    continue
                val = row[col]
                video_bytes = None

                if isinstance(val, bytes):
                    video_bytes = val
                elif isinstance(val, dict):
                    if 'bytes' in val and val['bytes'] is not None:
                        video_bytes = val['bytes']
                    if 'path' in val:
                        entry["original_path"] = val['path']

                if video_bytes and len(video_bytes) > 100:
                    label_tag = entry.get("label", "unk")
                    fname = f"{split_name}_{total:05d}_{label_tag}.mp4"
                    fpath = os.path.join(split_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(video_bytes)
                    entry["path"] = fpath
                    entry["filename"] = fname
                    entry["size_bytes"] = len(video_bytes)
                    video_saved = True
                    break

            if not video_saved:
                entry["path"] = ""
                entry["saved"] = False
            else:
                entry["saved"] = True

            samples.append(entry)
            total += 1

            if total % 100 == 0:
                labels = Counter(s.get("label", "unk") for s in samples)
                print(f"  Processed {total} samples... {dict(labels)}")

        if max_samples and total >= max_samples:
            break

    print(f"  Split '{split_name}' done: {total} samples")
    return samples


def verify_videos(samples, n_check=5):
    """Verify saved videos are valid using OpenCV."""
    import cv2

    valid = 0
    checked = 0
    for s in samples[:n_check]:
        path = s.get("path", "")
        if not path or not os.path.exists(path):
            continue
        checked += 1
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, frame = cap.read()
            cap.release()
            if ret and frames > 0:
                valid += 1
                print(f"  ✅ {s['filename']}: {frames:.0f}f, {w}x{h}, {fps:.1f}fps, "
                      f"label={s.get('label','?')}")
            else:
                print(f"  ❌ {s['filename']}: can't read frames")
        else:
            cap.release()
            print(f"  ❌ {s['filename']}: can't open")

    print(f"  Verified: {valid}/{checked}")
    return valid


def save_splits(all_samples):
    """Save split metadata as JSON."""
    splits = {}
    for split_name in ["train", "validation", "test"]:
        split_data = [s for s in all_samples if s["split"] == split_name]
        # Map validation → val
        save_name = "val" if split_name == "validation" else split_name
        splits[save_name] = split_data

        path = os.path.join(SPLITS_DIR, f"{save_name}.json")
        # Clean for JSON
        clean = []
        for s in split_data:
            c = {}
            for k, v in s.items():
                if isinstance(v, (np.integer,)):
                    c[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    c[k] = float(v)
                elif isinstance(v, (str, int, float, bool, type(None))):
                    c[k] = v
            clean.append(c)

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"  Saved {save_name}: {len(clean)} samples → {path}")

    # Check subject overlap
    for n1 in splits:
        for n2 in splits:
            if n1 >= n2:
                continue
            s1 = set(s.get("subject_id", "") for s in splits[n1])
            s2 = set(s.get("subject_id", "") for s in splits[n2])
            overlap = s1 & s2
            status = f"⚠️  {len(overlap)} overlapping" if overlap else "✅ disjoint"
            print(f"  {n1} ↔ {n2}: {status}")

    return splits


if __name__ == "__main__":
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("FakeAVCeleb Dataset Acquisition (Parquet-based)")
    print("=" * 70)

    # Phase 1: Get parquet file listing
    print("\n[Phase 1] Listing parquet files...")
    try:
        pq_files = get_parquet_urls()
    except Exception as e:
        print(f"  Failed to list files: {e}")
        print("  Trying alternative approach...")
        pq_files = None

    if not pq_files:
        # Fallback: use datasets library with decode disabled
        print("\n[Fallback] Using datasets library with video decoding disabled...")
        from datasets import load_dataset

        all_samples = []
        for split_name in ['train', 'validation', 'test']:
            print(f"\n  Loading split: {split_name}")
            try:
                ds = load_dataset(
                    HF_DATASET,
                    split=split_name,
                    streaming=True,
                )
                # Disable automatic decoding
                ds = ds.cast_column("video", None) if "video" in ds.column_names else ds
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
                continue
        sys.exit(1)

    # Phase 2: Download test batch
    print("\n[Phase 2] Downloading test batch (30 per split)...")
    all_samples = []
    for split_name, paths in pq_files.items():
        samples = download_and_process_split(split_name, paths, max_samples=30)
        all_samples.extend(samples)

    # Phase 3: Verify
    print("\n[Phase 3] Verifying videos...")
    for split in ["train", "validation", "test"]:
        split_samples = [s for s in all_samples if s["split"] == split and s.get("saved")]
        if split_samples:
            print(f"\n  --- {split} ---")
            verify_videos(split_samples)

    # Phase 4: Stats
    print("\n[Phase 4] Dataset statistics...")
    labels = Counter(s.get("label", "unk") for s in all_samples)
    methods = Counter(s.get("method", "none") for s in all_samples)
    subjects = set(s.get("subject_id", "") for s in all_samples)
    saved = sum(1 for s in all_samples if s.get("saved"))
    print(f"  Total: {len(all_samples)}, Saved: {saved}")
    print(f"  Labels: {dict(labels)}")
    print(f"  Methods: {dict(methods)}")
    print(f"  Subjects: {len(subjects)}")

    # Phase 5: Save splits
    print("\n[Phase 5] Saving splits...")
    save_splits(all_samples)

    print("\n" + "=" * 70)
    print("✅ Test batch complete!")
    print("=" * 70)
    print("\nTo download FULL dataset, re-run with max_samples=None")
