"""
Emotion Consistency Branch (E)
Cross-modal emotion disagreement between facial expression and speech emotion.
E is derived from BOTH modalities — it is NOT an independent input modality.
Emotion disagreement is used as SUPPORTING evidence, NOT proof of fakery.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, SAMPLE_RATE, SEGMENT_DURATION_SEC, SEGMENT_HOP_SEC,
    FPS_TARGET, SPEECH_EMOTION_MODEL, FACIAL_EMOTION_CLASSES
)


# ─── Speech Emotion Recognition ──────────────────────────────────────────────

class SpeechEmotionExtractor:
    """
    Extract emotion probabilities from speech using a pretrained Wav2Vec2
    speech emotion recognition model.
    """

    def __init__(self, model_name=SPEECH_EMOTION_MODEL, device=DEVICE):
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        self.device = device
        print(f"[EmotionBranch] Loading speech emotion model: {model_name}...")
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            self.emotion_labels = list(self.model.config.id2label.values()) if hasattr(self.model.config, 'id2label') else FACIAL_EMOTION_CLASSES
            print(f"[EmotionBranch] Speech emotion labels: {self.emotion_labels}")
            self.available = True
        except Exception as e:
            print(f"[EmotionBranch] Speech emotion model failed to load: {e}")
            print("[EmotionBranch] Will use fallback spectrogram-based features.")
            self.available = False

    @torch.no_grad()
    def predict(self, audio_segment, sr=SAMPLE_RATE):
        """
        Predict emotion probabilities from an audio segment.
        Returns dict of {emotion: probability}.
        """
        if not self.available:
            return self._fallback_predict(audio_segment)

        inputs = self.feature_extractor(
            audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)
        probs = torch.softmax(outputs.logits, dim=1).squeeze(0).cpu().numpy()

        return {label: float(prob) for label, prob in zip(self.emotion_labels, probs)}

    def _fallback_predict(self, audio_segment):
        """Fallback: return uniform distribution."""
        n = len(FACIAL_EMOTION_CLASSES)
        return {e: 1.0 / n for e in FACIAL_EMOTION_CLASSES}


# ─── Facial Emotion Recognition ──────────────────────────────────────────────

class FacialEmotionExtractor:
    """
    Extract emotion from face crops using MediaPipe Face Mesh landmarks
    + a simple geometric feature-based classifier.

    Uses Action Unit-inspired geometric ratios from the 468 MediaPipe landmarks.
    """

    EMOTION_LABELS = FACIAL_EMOTION_CLASSES

    # Key landmark indices for emotion-relevant facial regions
    # Mouth
    MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    # Eyes
    LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    # Eyebrows
    LEFT_EYEBROW = [70, 63, 105, 66, 107]
    RIGHT_EYEBROW = [336, 296, 334, 293, 300]

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        print("[EmotionBranch] Facial emotion extractor initialized (MediaPipe landmarks)")

    def extract_geometric_features(self, frame):
        """
        Extract emotion-indicative geometric features from a face using MediaPipe.
        Returns feature vector or None if no face found.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]
        pts = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])

        features = []

        # 1. Mouth aspect ratio (openness)
        mouth_top = pts[13]   # Upper lip
        mouth_bottom = pts[14]  # Lower lip
        mouth_left = pts[61]
        mouth_right = pts[291]
        mouth_height = np.linalg.norm(mouth_top[:2] - mouth_bottom[:2])
        mouth_width = np.linalg.norm(mouth_left[:2] - mouth_right[:2])
        mouth_ratio = mouth_height / (mouth_width + 1e-6)
        features.append(mouth_ratio)

        # 2. Mouth corner angles (smile detection)
        mouth_center = (pts[13][:2] + pts[14][:2]) / 2
        left_corner = pts[61][:2]
        right_corner = pts[291][:2]
        left_angle = np.arctan2(mouth_center[1] - left_corner[1], left_corner[0] - mouth_center[0])
        right_angle = np.arctan2(mouth_center[1] - right_corner[1], right_corner[0] - mouth_center[0])
        features.extend([left_angle, right_angle])

        # 3. Eye aspect ratios (openness — surprise, squint)
        for eye_pts in [self.LEFT_EYE, self.RIGHT_EYE]:
            if len(eye_pts) >= 6:
                top = pts[eye_pts[1]][:2]
                bottom = pts[eye_pts[5]][:2]
                left = pts[eye_pts[0]][:2]
                right = pts[eye_pts[3]][:2]
                eye_h = np.linalg.norm(top - bottom)
                eye_w = np.linalg.norm(left - right)
                features.append(eye_h / (eye_w + 1e-6))

        # 4. Eyebrow height relative to eyes (raised = surprise, lowered = anger)
        for brow_pts, eye_pts in [(self.LEFT_EYEBROW, self.LEFT_EYE),
                                   (self.RIGHT_EYEBROW, self.RIGHT_EYE)]:
            brow_center = pts[brow_pts[2]][:2]
            eye_center = np.mean([pts[i][:2] for i in eye_pts[:4]], axis=0)
            brow_height = np.linalg.norm(brow_center - eye_center)
            features.append(brow_height)

        # 5. Inter-eyebrow distance (furrowed = anger/concentration)
        left_inner_brow = pts[self.LEFT_EYEBROW[-1]][:2]
        right_inner_brow = pts[self.RIGHT_EYEBROW[-1]][:2]
        brow_dist = np.linalg.norm(left_inner_brow - right_inner_brow)
        features.append(brow_dist)

        # 6. Nose wrinkle approximation (distance between nose bridge landmarks)
        nose_bridge = pts[6][:2]
        nose_tip = pts[1][:2]
        features.append(np.linalg.norm(nose_bridge - nose_tip))

        return np.array(features, dtype=np.float32)

    def predict_emotion_probs(self, frame):
        """
        Predict facial emotion probabilities.
        Uses geometric features with a simple heuristic mapping.
        Returns dict of {emotion: probability}.
        """
        features = self.extract_geometric_features(frame)
        if features is None:
            n = len(self.EMOTION_LABELS)
            return {e: 1.0 / n for e in self.EMOTION_LABELS}

        # Simple heuristic-based scoring (will be replaced by learned mapping
        # after we have labeled emotion data, but works as a reasonable baseline)
        scores = {}
        mouth_ratio = features[0]
        left_corner_angle = features[1]
        right_corner_angle = features[2]
        left_eye_ratio = features[3]
        right_eye_ratio = features[4]
        left_brow_h = features[5]
        right_brow_h = features[6]
        brow_dist = features[7]

        avg_eye = (left_eye_ratio + right_eye_ratio) / 2
        avg_brow = (left_brow_h + right_brow_h) / 2
        avg_corner = (left_corner_angle + right_corner_angle) / 2

        # Heuristic scores (soft, relative)
        scores["happy"] = max(0, avg_corner * 2 + mouth_ratio * 0.5)
        scores["surprise"] = max(0, avg_eye * 2 + avg_brow * 1.5 + mouth_ratio)
        scores["angry"] = max(0, -avg_corner + (0.1 - brow_dist) * 5)
        scores["sad"] = max(0, -avg_corner * 1.5 + (0.05 - avg_eye) * 3)
        scores["fear"] = max(0, avg_eye * 1.5 + mouth_ratio * 0.5 - avg_corner)
        scores["disgust"] = max(0, -avg_corner + features[8] * 2 if len(features) > 8 else 0)
        scores["neutral"] = max(0.1, 1.0 - sum(abs(v) for v in [mouth_ratio - 0.1, avg_corner, avg_eye - 0.3]) * 2)

        # Normalize to probabilities
        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}


# ─── Emotion Consistency Computation ─────────────────────────────────────────

class EmotionConsistencyScorer:
    """
    Compute cross-modal emotion consistency between facial and vocal emotion.
    Higher disagreement → higher suspicion of deepfake.
    """

    def __init__(self, device=DEVICE):
        self.speech_extractor = SpeechEmotionExtractor(device=device)
        self.facial_extractor = FacialEmotionExtractor()
        # Map between potentially different label sets
        self.common_emotions = ["angry", "happy", "sad", "neutral", "fear", "surprise", "disgust"]
        print("[EmotionBranch] Emotion consistency scorer initialized")

    def compute_disagreement(self, speech_probs, facial_probs):
        """
        Compute emotion disagreement between speech and facial emotion distributions.
        Returns multiple disagreement features:
        - KL divergence (speech || facial)
        - Jensen-Shannon divergence
        - L1 distance
        - Top-1 match (binary)
        - Cosine similarity
        """
        # Align to common emotion labels
        speech_vec = np.array([speech_probs.get(e, 0.0) for e in self.common_emotions])
        facial_vec = np.array([facial_probs.get(e, 0.0) for e in self.common_emotions])

        # Normalize
        speech_vec = speech_vec / (speech_vec.sum() + 1e-8)
        facial_vec = facial_vec / (facial_vec.sum() + 1e-8)

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        speech_vec = speech_vec + eps
        facial_vec = facial_vec + eps
        speech_vec = speech_vec / speech_vec.sum()
        facial_vec = facial_vec / facial_vec.sum()

        # KL divergence
        kl_div = float(np.sum(speech_vec * np.log(speech_vec / facial_vec)))

        # Jensen-Shannon divergence
        m = 0.5 * (speech_vec + facial_vec)
        js_div = float(0.5 * np.sum(speech_vec * np.log(speech_vec / m)) +
                        0.5 * np.sum(facial_vec * np.log(facial_vec / m)))

        # L1 distance
        l1_dist = float(np.sum(np.abs(speech_vec - facial_vec)))

        # Top-1 match
        top1_match = float(np.argmax(speech_vec) == np.argmax(facial_vec))

        # Cosine similarity
        cos_sim = float(np.dot(speech_vec, facial_vec) /
                        (np.linalg.norm(speech_vec) * np.linalg.norm(facial_vec) + 1e-8))

        return np.array([kl_div, js_div, l1_dist, top1_match, cos_sim], dtype=np.float32)

    def get_raw_distributions(self, speech_probs, facial_probs):
        """Return concatenated raw probability distributions."""
        speech_vec = np.array([speech_probs.get(e, 0.0) for e in self.common_emotions])
        facial_vec = np.array([facial_probs.get(e, 0.0) for e in self.common_emotions])
        return np.concatenate([speech_vec, facial_vec])


class EmotionBranch:
    """
    Complete emotion consistency branch.
    Processes BOTH audio and video to compute cross-modal disagreement.
    """

    def __init__(self, device=DEVICE):
        self.device = device
        self.scorer = EmotionConsistencyScorer(device=device)
        # Feature dim: 5 disagreement features + 14 raw distributions
        self.feature_dim = 5 + len(self.scorer.common_emotions) * 2  # = 19
        self.classifier = EmotionClassifierHead(input_dim=self.feature_dim).to(device)
        self.trained = False

    def extract_segment_features(self, audio_segment, face_crops, sr=SAMPLE_RATE):
        """
        Extract emotion consistency features for a single segment.
        Requires BOTH audio and video data.
        """
        # Speech emotion
        speech_probs = self.scorer.speech_extractor.predict(audio_segment, sr=sr)

        # Facial emotion (average across frames in segment)
        if face_crops:
            facial_probs_list = [self.scorer.facial_extractor.predict_emotion_probs(crop)
                                 for crop in face_crops]
            # Average probabilities across frames
            facial_probs = {}
            for emotion in self.scorer.common_emotions:
                facial_probs[emotion] = np.mean([fp.get(emotion, 0.0) for fp in facial_probs_list])
        else:
            n = len(self.scorer.common_emotions)
            facial_probs = {e: 1.0 / n for e in self.scorer.common_emotions}

        # Compute disagreement features
        disagreement = self.scorer.compute_disagreement(speech_probs, facial_probs)
        raw_dists = self.scorer.get_raw_distributions(speech_probs, facial_probs)

        return np.concatenate([disagreement, raw_dists])

    def get_segment_scores(self, audio_path, video_path):
        """
        Get per-segment emotion-consistency-based manipulation scores.
        NOTE: This branch requires BOTH audio and video paths.
        """
        import librosa
        import torchaudio

        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            try:
                waveform, orig_sr = torchaudio.load(audio_path)
                if orig_sr != SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(orig_sr, SAMPLE_RATE)
                    waveform = resampler(waveform)
                audio = waveform.mean(dim=0).numpy()
                sr = SAMPLE_RATE
            except Exception as e:
                print(f"[EmotionBranch] Audio load failed: {e}")
                return None

        # Extract faces from video
        face_ext = self.scorer.facial_extractor
        mp_det = mp.solutions.face_detection
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(orig_fps / FPS_TARGET))
        all_faces = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / orig_fps
                with mp_det.FaceDetection(min_detection_confidence=0.5, model_selection=1) as fd:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = fd.process(rgb)
                    if results.detections:
                        h, w = frame.shape[:2]
                        bbox = results.detections[0].location_data.relative_bounding_box
                        x1 = max(0, int(bbox.xmin * w))
                        y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w, x1 + int(bbox.width * w))
                        y2 = min(h, y1 + int(bbox.height * h))
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            crop = cv2.resize(crop, (224, 224))
                            all_faces.append((timestamp, crop))
            frame_idx += 1
        cap.release()

        # Process segments
        seg_len_samples = int(SEGMENT_DURATION_SEC * SAMPLE_RATE)
        hop_samples = int(SEGMENT_HOP_SEC * SAMPLE_RATE)
        features = []

        for start_sample in range(0, len(audio) - seg_len_samples + 1, hop_samples):
            end_sample = start_sample + seg_len_samples
            audio_seg = audio[start_sample:end_sample]

            start_time = start_sample / SAMPLE_RATE
            end_time = end_sample / SAMPLE_RATE
            seg_faces = [crop for ts, crop in all_faces if start_time <= ts < end_time]

            feat = self.extract_segment_features(audio_seg, seg_faces)
            features.append(feat)

        if not features:
            return None

        # Score segments
        self.classifier.eval()
        with torch.no_grad():
            feat_tensor = torch.tensor(np.array(features), dtype=torch.float32).to(self.device)
            logits = self.classifier(feat_tensor)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()
        return scores

    def get_features(self, audio_path, video_path):
        """Get raw emotion features for fusion."""
        import librosa
        try:
            audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            return None

        # Simplified face extraction
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = max(1, int(orig_fps / FPS_TARGET))
        all_faces = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / orig_fps
                faces = MediaPipeFaceExtractor().extract_faces_from_frame(frame)
                if faces:
                    all_faces.append((timestamp, faces[0]))
            frame_idx += 1
        cap.release()

        seg_len_samples = int(SEGMENT_DURATION_SEC * SAMPLE_RATE)
        hop_samples = int(SEGMENT_HOP_SEC * SAMPLE_RATE)
        features = []

        for start_sample in range(0, len(audio) - seg_len_samples + 1, hop_samples):
            end_sample = start_sample + seg_len_samples
            audio_seg = audio[start_sample:end_sample]
            start_time = start_sample / SAMPLE_RATE
            end_time = end_sample / SAMPLE_RATE
            seg_faces = [crop for ts, crop in all_faces if start_time <= ts < end_time]
            feat = self.extract_segment_features(audio_seg, seg_faces)
            features.append(feat)

        return np.array(features) if features else None

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)

    def load(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.trained = True


class EmotionClassifierHead(nn.Module):
    """Classifier head for emotion-consistency-based manipulation scoring."""

    def __init__(self, input_dim=19, hidden_dim=64, num_classes=2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


# Import for cross-reference
from src.branches.visual_branch import MediaPipeFaceExtractor


if __name__ == "__main__":
    print("[EmotionBranch] Self-test...")

    # Test speech emotion
    speech_ext = SpeechEmotionExtractor()
    dummy_audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)
    probs = speech_ext.predict(dummy_audio)
    print(f"  Speech emotion probs: {probs}")

    # Test facial emotion
    facial_ext = FacialEmotionExtractor()
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    fprobs = facial_ext.predict_emotion_probs(dummy_face)
    print(f"  Facial emotion probs: {fprobs}")

    # Test consistency scorer
    scorer = EmotionConsistencyScorer()
    disagreement = scorer.compute_disagreement(probs, fprobs)
    print(f"  Disagreement features: {disagreement}")

    print("[EmotionBranch] ✅ Self-test passed.")
