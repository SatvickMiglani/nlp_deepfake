"""
Multimodal Deepfake Detection - Scalable Feature Extraction Pipeline
Uses MediaPipe FaceLandmarker for visual landmarks & 52-AU facial blendshapes,
EfficientNet-B0 for visual forensics, acoustic CNN for audio forensics, and cross-modal
emotion divergence for the E-branch.

Features:
- Routes all Hugging Face downloads to D:\\deepfake\\hf_cache (378 GB free space)
- Automatic checkpointing & resumption (skips already extracted clips)
- Supports extracting all available videos, targeted 2,000 paper benchmark, or balanced mode
"""
import os
import sys

# Ensure HF cache is strictly on D: drive to protect C: drive space
os.environ["HF_HOME"] = r"D:\deepfake\hf_cache"
os.environ["TORCH_HOME"] = r"D:\deepfake\models"

import argparse
import time
import json
import urllib.request
from collections import Counter, defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from huggingface_hub import hf_hub_download, HfApi
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, SCORE_CACHE_DIR, RANDOM_SEED
)

MODEL_DIR = r"D:\deepfake\models"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCORE_CACHE_DIR, exist_ok=True)
LANDMARKER_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def ensure_mediapipe_model():
    if not os.path.exists(LANDMARKER_PATH):
        print(f"[Setup] Downloading MediaPipe face_landmarker.task to {LANDMARKER_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, LANDMARKER_PATH)
        print("[Setup] Download complete.")


class MediaPipeFeatureExtractor:
    def __init__(self):
        ensure_mediapipe_model()
        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=LANDMARKER_PATH),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract_from_rgb(self, rgb_img):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        res = self.landmarker.detect(mp_image)
        blendshapes = {}
        if res.face_blendshapes:
            for b in res.face_blendshapes[0]:
                blendshapes[b.category_name] = float(b.score)
        landmarks = None
        if res.face_landmarks:
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in res.face_landmarks[0]], dtype=np.float32)
        return blendshapes, landmarks


def blendshapes_to_emotion_distribution(bs_dict):
    if not bs_dict:
        return np.ones(len(EMOTIONS), dtype=np.float32) / len(EMOTIONS)

    happy = (bs_dict.get("mouthSmileLeft", 0) + bs_dict.get("mouthSmileRight", 0)) * 0.5 + \
            (bs_dict.get("cheekSquintLeft", 0) + bs_dict.get("cheekSquintRight", 0)) * 0.25

    angry = (bs_dict.get("browDownLeft", 0) + bs_dict.get("browDownRight", 0)) * 0.5 + \
            bs_dict.get("jawForward", 0) * 0.2 + bs_dict.get("mouthPressLeft", 0) * 0.15

    sad = bs_dict.get("browInnerUp", 0) * 0.4 + \
          (bs_dict.get("mouthFrownLeft", 0) + bs_dict.get("mouthFrownRight", 0)) * 0.4 + \
          bs_dict.get("mouthShrugLower", 0) * 0.2

    surprise = bs_dict.get("jawOpen", 0) * 0.4 + \
               (bs_dict.get("eyeWideLeft", 0) + bs_dict.get("eyeWideRight", 0)) * 0.3 + \
               (bs_dict.get("browOuterUpLeft", 0) + bs_dict.get("browOuterUpRight", 0)) * 0.3

    fear = bs_dict.get("browInnerUp", 0) * 0.35 + \
           (bs_dict.get("eyeWideLeft", 0) + bs_dict.get("eyeWideRight", 0)) * 0.35 + \
           (bs_dict.get("mouthStretchLeft", 0) + bs_dict.get("mouthStretchRight", 0)) * 0.3

    disgust = (bs_dict.get("noseSneerLeft", 0) + bs_dict.get("noseSneerRight", 0)) * 0.6 + \
              (bs_dict.get("mouthUpperUpLeft", 0) + bs_dict.get("mouthUpperUpRight", 0)) * 0.4

    active_sum = happy + angry + sad + surprise + fear + disgust
    neutral = max(0.05, 1.0 - active_sum * 1.5)

    scores = np.array([angry, disgust, fear, happy, sad, surprise, neutral], dtype=np.float32)
    scores = np.maximum(scores, 1e-4)
    probs = scores / np.sum(scores)
    return probs


class AudioCNNEncoder(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, feat_dim)
        self.emotion_head = nn.Linear(feat_dim, len(EMOTIONS))

    def forward(self, x):
        feats = self.conv(x).view(x.size(0), -1)
        emb = self.fc(feats)
        emo_logits = self.emotion_head(emb)
        return emb, torch.softmax(emo_logits, dim=-1)


class VisualCNNEncoder:
    def __init__(self, device=DEVICE):
        self.device = device
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.model.classifier = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def extract_batch(self, rgb_frames):
        tensors = torch.stack([self.transform(f) for f in rgb_frames]).to(self.device)
        feats = self.model(tensors)
        return feats.cpu().numpy()


def compute_divergences(p_audio, p_video):
    eps = 1e-7
    p = np.clip(p_audio, eps, 1.0)
    q = np.clip(p_video, eps, 1.0)
    p = p / np.sum(p)
    q = q / np.sum(q)

    kl_pq = np.sum(p * np.log(p / q))
    kl_qp = np.sum(q * np.log(q / p))
    sym_kl = float(0.5 * (kl_pq + kl_qp))

    m = 0.5 * (p + q)
    js_div = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    l1_dist = float(np.sum(np.abs(p - q)))

    cos_sim = float(np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + eps))
    cos_dist = float(1.0 - cos_sim)

    top1_match = float(np.argmax(p) == np.argmax(q))

    return np.array([sym_kl, js_div, l1_dist, cos_dist, top1_match], dtype=np.float32)


def run_extraction(target_clips=None, max_shards=None, resume=True, device=DEVICE):
    print("=" * 80)
    print("MULTIMODAL DEEPFAKE DETECTION — SCALE FEATURE EXTRACTION")
    print("=" * 80)
    print(f"Target clips: {'ALL available' if target_clips is None else target_clips}")
    print(f"Device: {device}")
    print(f"Hugging Face Cache: {os.environ['HF_HOME']}")

    # 1. Load Audio Manifest
    print("\n[1/4] Loading audio manifest...")
    manifest_p = hf_hub_download(
        repo_id="advy/fakeavceleb-detect",
        filename="preprocessing_cache/audio_mfcc_shards_v2/manifest_v2.json"
    )
    with open(manifest_p) as f:
        audio_manifest = json.load(f)
    print(f"Audio manifest contains {len(audio_manifest)} videos across all 4 categories.")

    # 2. Query All Available Face Shards on HF Hub
    api = HfApi()
    face_files = list(api.list_repo_tree('advy/fakeavceleb-detect', path_in_repo='preprocessing_cache/face_crops_shards', recursive=False))
    shard_names = sorted([f.path.split('/')[-1] for f in face_files if f.path.endswith('.pt')])
    print(f"Found {len(shard_names)} face shards on Hugging Face Hub.")

    if max_shards:
        shard_names = shard_names[:max_shards]
        print(f"Processing first {len(shard_names)} shards.")

    # Prioritize balanced shards first so all 4 categories are well-represented early
    priority_shards = ["shard_00000.pt", "shard_00001.pt", "shard_00004.pt", "shard_00005.pt",
                       "shard_00006.pt", "shard_00007.pt", "shard_00009.pt", "shard_00010.pt",
                       "shard_00013.pt", "shard_00014.pt", "shard_00040.pt", "shard_00160.pt"]
    ordered_shards = [s for s in priority_shards if s in shard_names] + [s for s in shard_names if s not in priority_shards]

    # 3. Resume Checkpoint if available
    save_path = os.path.join(SCORE_CACHE_DIR, "extracted_features.pt")
    extracted_samples = []
    processed_vids = set()

    if resume and os.path.exists(save_path):
        try:
            extracted_samples = torch.load(save_path, map_location="cpu", weights_only=False)
            processed_vids = set(s["video_id"] for s in extracted_samples)
            print(f"[Resume] Found existing checkpoint with {len(extracted_samples)} videos. Resuming...")
        except Exception as e:
            print(f"[Warning] Could not load checkpoint ({e}), starting fresh.")
            extracted_samples = []
            processed_vids = set()

    if target_clips and len(extracted_samples) >= target_clips:
        print(f"[Notice] Already have {len(extracted_samples)} extracted videos (>= target {target_clips}). Done!")
        return save_path

    # 4. Initialize Feature Extractors
    print("\n[2/4] Initializing MediaPipe Tasks FaceLandmarker...")
    mp_extractor = MediaPipeFeatureExtractor()

    print("[3/4] Initializing Visual CNN Encoder (EfficientNet-B0)...")
    vis_encoder = VisualCNNEncoder(device=device)

    print("[4/4] Initializing Audio CNN Encoder...")
    audio_encoder = AudioCNNEncoder().to(device)
    audio_encoder.eval()

    total_processed = len(extracted_samples)
    t0 = time.time()
    audio_shard_cache = {}

    print(f"\n[Extraction Starting] Processing shards...")

    for shard_idx, f_shard_name in enumerate(ordered_shards):
        if target_clips and total_processed >= target_clips:
            print(f"\n---> Reached target of {target_clips} videos. Stopping extraction.")
            break

        print(f"\n---> [{shard_idx+1}/{len(ordered_shards)}] Face Shard: {f_shard_name} (Total Extracted: {total_processed})...")

        try:
            face_p = hf_hub_download(
                repo_id="advy/fakeavceleb-detect",
                filename=f"preprocessing_cache/face_crops_shards/{f_shard_name}"
            )
            f_data = torch.load(face_p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  Warning: failed to download {f_shard_name}: {e}. Skipping.")
            continue

        shard_new_count = 0
        for vid, f_item in f_data.items():
            if target_clips and total_processed >= target_clips:
                break

            if vid in processed_vids or vid not in audio_manifest:
                continue

            a_shard_name = audio_manifest[vid]

            # Cache audio shard in memory (evicting older ones to prevent RAM bloat)
            if a_shard_name not in audio_shard_cache:
                if len(audio_shard_cache) > 4:
                    # Keep at most 4 audio shards in RAM
                    oldest = list(audio_shard_cache.keys())[0]
                    del audio_shard_cache[oldest]
                try:
                    a_shard_p = hf_hub_download(
                        repo_id="advy/fakeavceleb-detect",
                        filename=f"preprocessing_cache/audio_mfcc_shards_v2/{a_shard_name}"
                    )
                    audio_shard_cache[a_shard_name] = torch.load(a_shard_p, map_location="cpu", weights_only=False)
                except Exception as e:
                    continue

            a_shard_data = audio_shard_cache[a_shard_name]
            if vid not in a_shard_data:
                continue

            a_item = a_shard_data[vid]

            # 1. Decode Face Crops & Run MediaPipe
            jpegs = f_item.get("jpegs", [])
            if not jpegs:
                continue

            rgb_frames = []
            frame_blendshapes = []
            for j in jpegs:
                arr = np.frombuffer(j, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb_frames.append(rgb)
                    bs, lm = mp_extractor.extract_from_rgb(rgb)
                    frame_blendshapes.append(bs)

            if not rgb_frames:
                continue

            # Visual Forensic Feature V (1280,)
            v_feats_per_frame = vis_encoder.extract_batch(rgb_frames)
            v_clip_feat = np.mean(v_feats_per_frame, axis=0)

            # Facial emotion distribution (7,)
            facial_emo_per_frame = [blendshapes_to_emotion_distribution(bs) for bs in frame_blendshapes]
            facial_emo_mean = np.mean(facial_emo_per_frame, axis=0)

            # 2. Audio MFCC features (128,)
            mfccs = a_item.get("mfccs", [])
            if not mfccs:
                continue

            audio_tensors = torch.stack([m if isinstance(m, torch.Tensor) else torch.tensor(m) for m in mfccs]).float().to(device)
            with torch.no_grad():
                a_win_feats, a_win_emos = audio_encoder(audio_tensors)
                a_win_feats = a_win_feats.cpu().numpy()
                a_win_emos = a_win_emos.cpu().numpy()

            a_clip_feat = np.mean(a_win_feats, axis=0)
            audio_emo_mean = np.mean(a_win_emos, axis=0)

            # 3. Emotion Consistency Divergences E (19,)
            divergences_per_window = []
            for w_idx in range(len(a_win_emos)):
                f_idx = min(int(w_idx * len(facial_emo_per_frame) / len(a_win_emos)), len(facial_emo_per_frame)-1)
                divs = compute_divergences(a_win_emos[w_idx], facial_emo_per_frame[f_idx])
                divergences_per_window.append(divs)

            divergences_per_window = np.array(divergences_per_window)
            e_clip_feat = np.mean(divergences_per_window, axis=0)
            e_full_feat = np.concatenate([e_clip_feat, audio_emo_mean, facial_emo_mean])

            # Labels & metadata
            cat_name = vid.split("__")[0]
            v_label = int(f_item.get("visual_label", 0))
            a_label = int(a_item.get("label", 0))
            is_fake = int("Fake" in cat_name)

            parts = vid.split("__")
            subject_id = parts[1].split("_")[0] if len(parts) > 1 else f"subj_{total_processed % 100}"

            sample_entry = {
                "video_id": vid,
                "subject_id": subject_id,
                "category": cat_name,
                "visual_label": v_label,
                "audio_label": a_label,
                "label": is_fake,
                "v_feat": v_clip_feat,
                "a_feat": a_clip_feat,
                "e_feat": e_full_feat,
                "window_divergences": divergences_per_window,
                "n_frames": len(rgb_frames),
                "n_windows": len(mfccs)
            }
            extracted_samples.append(sample_entry)
            processed_vids.add(vid)
            total_processed += 1
            shard_new_count += 1

            if total_processed % 50 == 0:
                elapsed_now = time.time() - t0
                speed = total_processed / max(1, elapsed_now)
                print(f"    Progress: {total_processed} videos extracted ({speed:.2f} vid/sec) | [Current: {cat_name}]")

        # Save checkpoint after each face shard
        if shard_new_count > 0:
            torch.save(extracted_samples, save_path)
            print(f"  [Checkpoint] Saved {len(extracted_samples)} videos to {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    elapsed = time.time() - t0
    cats = Counter(s["category"] for s in extracted_samples)
    labels = Counter(s["label"] for s in extracted_samples)
    print("\n" + "=" * 80)
    print(f"[EXTRACTION COMPLETE] Processed {len(extracted_samples)} videos in {elapsed/60:.2f} min ({total_processed/max(1,elapsed):.2f} videos/sec).")
    print(f"Categories Breakdown: {dict(cats)}")
    print(f"Binary Fake vs Real: {dict(labels)}")
    print(f"Output File: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")
    print("=" * 80)
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale Multimodal Feature Extraction")
    parser.add_argument("--target_clips", type=int, default=None, help="Target total videos to extract (e.g. 2000 to match paper benchmark)")
    parser.add_argument("--max_shards", type=int, default=None, help="Max number of shards to process")
    parser.add_argument("--all", action="store_true", help="Extract all available videos in the dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--no_resume", action="store_true", help="Do not resume existing checkpoint")
    args = parser.parse_args()

    target = None
    if not args.all:
        target = args.target_clips if args.target_clips is not None else 2144

    run_extraction(target_clips=target, max_shards=args.max_shards, resume=not args.no_resume, device=args.device)
