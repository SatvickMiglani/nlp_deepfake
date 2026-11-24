"""
Visual Forensic Branch (V)
Uses MediaPipe for face detection/mesh and a frozen CNN for deepfake scoring.
MediaPipe replaces Haar Cascade for more robust face detection.
"""
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, FPS_TARGET, SEGMENT_DURATION_SEC, SEGMENT_HOP_SEC,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE, VISUAL_CLASSIFIER_HIDDEN
)


class MediaPipeFaceExtractor:
    """
    Extract face crops from video frames using MediaPipe Face Detection.
    This replaces Haar Cascade for more robust, orientation-invariant detection.
    """

    def __init__(self, min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE):
        self.mp_face_detection = mp.solutions.face_detection
        self.min_conf = min_detection_confidence
        print(f"[VisualBranch] MediaPipe Face Detection initialized (conf={min_detection_confidence})")

    def extract_faces_from_frame(self, frame, target_size=(224, 224)):
        """
        Detect and crop faces from a single frame using MediaPipe.
        Returns list of face crops (BGR, resized to target_size).
        """
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_conf,
            model_selection=1  # 1 = full-range model (better for varied distances)
        ) as face_detection:
            results = face_detection.process(rgb_frame)

        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)

                # Add margin (15%)
                margin_x = int(bw * 0.15)
                margin_y = int(bh * 0.15)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(w, x + bw + margin_x)
                y2 = min(h, y + bh + margin_y)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_crop = cv2.resize(face_crop, target_size)
                    faces.append(face_crop)

        return faces

    def extract_faces_from_video(self, video_path, fps=FPS_TARGET):
        """
        Extract face crops from a video at the target FPS.
        Returns list of (timestamp_sec, face_crop) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[VisualBranch] Failed to open video: {video_path}")
            return []

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps <= 0:
            orig_fps = 30
        frame_interval = max(1, int(orig_fps / fps))

        faces = []
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                timestamp = frame_idx / orig_fps
                detected = self.extract_faces_from_frame(frame)
                if detected:
                    faces.append((timestamp, detected[0]))  # Take largest/first face
            frame_idx += 1

        cap.release()
        return faces


class MediaPipeFaceMeshExtractor:
    """
    Extract face mesh landmarks using MediaPipe Face Mesh.
    Provides 468 3D landmarks per face — useful for geometric features.
    """

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        print("[VisualBranch] MediaPipe Face Mesh initialized")

    def extract_landmarks(self, frame):
        """Extract 468 face landmarks from a frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark])
            return coords  # Shape: (468, 3)
        return None


class VisualFeatureExtractor:
    """
    Extract visual features from face crops using a frozen pretrained CNN.
    Uses EfficientNet-B0 as the backbone.
    """

    def __init__(self, device=DEVICE):
        from torchvision import models, transforms
        self.device = device

        print("[VisualBranch] Loading EfficientNet-B0...")
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Remove classifier head, use features only
        self.model.classifier = nn.Identity()
        self.model = self.model.to(device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.feature_dim = 1280  # EfficientNet-B0 output dim
        print(f"[VisualBranch] EfficientNet-B0 loaded (feature_dim={self.feature_dim})")

    @torch.no_grad()
    def extract_features(self, face_crop):
        """Extract features from a single face crop (BGR numpy array)."""
        # Convert BGR to RGB
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        features = self.model(tensor).squeeze(0).cpu().numpy()
        return features

    def extract_segment_features(self, face_crops):
        """
        Extract and average features across multiple face crops in a segment.
        Returns a single feature vector for the segment.
        """
        if not face_crops:
            return None
        features = [self.extract_features(crop) for crop in face_crops]
        return np.mean(features, axis=0)


class VisualClassifierHead(nn.Module):
    """Classifier head for visual forensic scoring."""

    def __init__(self, input_dim=1280, hidden_dim=VISUAL_CLASSIFIER_HIDDEN, num_classes=2):
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


class VisualBranch:
    """Complete visual forensic branch: MediaPipe face detection → CNN features → classifier."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.face_extractor = MediaPipeFaceExtractor()
        self.feature_extractor = VisualFeatureExtractor(device=device)
        self.classifier = VisualClassifierHead(
            input_dim=self.feature_extractor.feature_dim
        ).to(device)
        self.trained = False

    def get_segment_scores(self, video_path):
        """
        Get per-segment manipulation probability scores.
        Returns array of shape (n_segments,) with values in [0, 1].
        """
        # Extract faces
        face_data = self.face_extractor.extract_faces_from_video(video_path)
        if not face_data:
            return None

        # Group faces into segments
        seg_dur = SEGMENT_DURATION_SEC
        hop = SEGMENT_HOP_SEC
        max_time = face_data[-1][0]
        segments = []
        t = 0
        while t <= max_time:
            seg_faces = [crop for ts, crop in face_data if t <= ts < t + seg_dur]
            if seg_faces:
                feat = self.feature_extractor.extract_segment_features(seg_faces)
                if feat is not None:
                    segments.append(feat)
            t += hop

        if not segments:
            return None

        # Score segments
        self.classifier.eval()
        with torch.no_grad():
            feat_tensor = torch.tensor(np.array(segments), dtype=torch.float32).to(self.device)
            logits = self.classifier(feat_tensor)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1].cpu().numpy()
        return scores

    def get_features(self, video_path):
        """Get raw features for fusion."""
        face_data = self.face_extractor.extract_faces_from_video(video_path)
        if not face_data:
            return None

        seg_dur = SEGMENT_DURATION_SEC
        hop = SEGMENT_HOP_SEC
        max_time = face_data[-1][0]
        segments = []
        t = 0
        while t <= max_time:
            seg_faces = [crop for ts, crop in face_data if t <= ts < t + seg_dur]
            if seg_faces:
                feat = self.feature_extractor.extract_segment_features(seg_faces)
                if feat is not None:
                    segments.append(feat)
            t += hop
        return np.array(segments) if segments else None

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)
        print(f"[VisualBranch] Classifier saved to {path}")

    def load(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.trained = True
        print(f"[VisualBranch] Classifier loaded from {path}")


if __name__ == "__main__":
    print("[VisualBranch] Self-test...")

    # Test MediaPipe face detection on a synthetic image
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face_ext = MediaPipeFaceExtractor()
    faces = face_ext.extract_faces_from_frame(dummy_frame)
    print(f"  Faces detected in random noise: {len(faces)} (expected 0)")

    # Test feature extractor
    feat_ext = VisualFeatureExtractor()
    dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = feat_ext.extract_features(dummy_face)
    print(f"  Feature dim: {features.shape}")

    # Test face mesh
    mesh_ext = MediaPipeFaceMeshExtractor()
    landmarks = mesh_ext.extract_landmarks(dummy_frame)
    print(f"  Landmarks from random noise: {landmarks}")

    print("[VisualBranch] ✅ Self-test passed.")
