"""
Audio Forensic Branch (A)
Extracts audio features using Wav2Vec2 and scores manipulation probability.
Uses pretrained Wav2Vec2 as a frozen feature extractor + trainable classifier head.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, SAMPLE_RATE, SEGMENT_DURATION_SEC, SEGMENT_HOP_SEC,
    AUDIO_MODEL_NAME, AUDIO_CLASSIFIER_HIDDEN
)


class AudioFeatureExtractor:
    """Extract audio features from video files using Wav2Vec2."""

    def __init__(self, model_name=AUDIO_MODEL_NAME, device=DEVICE):
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.device = device
        print(f"[AudioBranch] Loading {model_name}...")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        print(f"[AudioBranch] Model loaded on {device}.")

    def load_audio(self, path, sr=SAMPLE_RATE):
        """Load audio from video or audio file."""
        try:
            # Try loading with torchaudio first
            waveform, orig_sr = torchaudio.load(path)
            if orig_sr != sr:
                resampler = torchaudio.transforms.Resample(orig_sr, sr)
                waveform = resampler(waveform)
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            return waveform.squeeze(0).numpy()
        except Exception:
            # Fallback to librosa
            try:
                audio, _ = librosa.load(path, sr=sr, mono=True)
                return audio
            except Exception as e:
                print(f"[AudioBranch] Failed to load audio from {path}: {e}")
                return None

    def segment_audio(self, audio, sr=SAMPLE_RATE):
        """Split audio into overlapping segments."""
        seg_len = int(SEGMENT_DURATION_SEC * sr)
        hop_len = int(SEGMENT_HOP_SEC * sr)
        segments = []
        for start in range(0, len(audio) - seg_len + 1, hop_len):
            segments.append(audio[start:start + seg_len])
        if not segments and len(audio) > 0:
            # Pad short audio
            padded = np.zeros(seg_len)
            padded[:len(audio)] = audio
            segments.append(padded)
        return segments

    @torch.no_grad()
    def extract_features(self, audio_segment, sr=SAMPLE_RATE):
        """Extract Wav2Vec2 features from a single segment."""
        inputs = self.feature_extractor(
            audio_segment, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)
        outputs = self.model(input_values)
        # Use mean of last hidden state as feature vector
        features = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        return features

    def extract_all_segments(self, audio_path):
        """Extract features for all segments in an audio file."""
        audio = self.load_audio(audio_path)
        if audio is None:
            return None
        segments = self.segment_audio(audio)
        features = []
        for seg in segments:
            feat = self.extract_features(seg)
            features.append(feat)
        return np.array(features)  # Shape: (n_segments, feature_dim)


class AudioClassifierHead(nn.Module):
    """Simple classifier head for audio forensic scoring."""

    def __init__(self, input_dim=768, hidden_dim=AUDIO_CLASSIFIER_HIDDEN, num_classes=2):
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


class AudioBranch:
    """Complete audio forensic branch: extraction + classification."""

    def __init__(self, device=DEVICE):
        self.device = device
        self.extractor = AudioFeatureExtractor(device=device)
        self.classifier = AudioClassifierHead().to(device)
        self.trained = False

    def get_segment_scores(self, audio_path):
        """
        Get per-segment manipulation probability scores.
        Returns array of shape (n_segments,) with values in [0, 1].
        """
        features = self.extractor.extract_all_segments(audio_path)
        if features is None:
            return None

        self.classifier.eval()
        with torch.no_grad():
            feat_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self.classifier(feat_tensor)
            probs = torch.softmax(logits, dim=1)
            # Return probability of fake (class 1)
            scores = probs[:, 1].cpu().numpy()
        return scores

    def get_features(self, audio_path):
        """Get raw features for fusion."""
        return self.extractor.extract_all_segments(audio_path)

    def save(self, path):
        torch.save(self.classifier.state_dict(), path)
        print(f"[AudioBranch] Classifier saved to {path}")

    def load(self, path):
        self.classifier.load_state_dict(torch.load(path, map_location=self.device))
        self.trained = True
        print(f"[AudioBranch] Classifier loaded from {path}")


# ─── Spectrogram-based alternative ───────────────────────────────────────────

class SpectrogramFeatureExtractor:
    """Alternative audio features using mel spectrograms (lighter weight)."""

    def __init__(self, sr=SAMPLE_RATE, n_mels=128, n_fft=1024, hop_length=512):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio_segment):
        """Extract mel spectrogram features."""
        mel = librosa.feature.melspectrogram(
            y=audio_segment, sr=self.sr,
            n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Flatten or summarize
        features = np.concatenate([
            mel_db.mean(axis=1),   # Mean across time
            mel_db.std(axis=1),    # Std across time
            mel_db.max(axis=1),    # Max across time
        ])
        return features

    def extract_from_file(self, path, sr=SAMPLE_RATE):
        """Extract features from file, segment by segment."""
        try:
            audio, _ = librosa.load(path, sr=sr, mono=True)
        except Exception as e:
            print(f"[SpectrogramExtractor] Failed: {e}")
            return None

        seg_len = int(SEGMENT_DURATION_SEC * sr)
        hop_len = int(SEGMENT_HOP_SEC * sr)
        features = []
        for start in range(0, len(audio) - seg_len + 1, hop_len):
            segment = audio[start:start + seg_len]
            features.append(self.extract(segment))
        if not features and len(audio) > 0:
            padded = np.zeros(seg_len)
            padded[:len(audio)] = audio
            features.append(self.extract(padded))
        return np.array(features) if features else None


if __name__ == "__main__":
    print("[AudioBranch] Self-test...")
    # Quick sanity check with a synthetic signal
    extractor = SpectrogramFeatureExtractor()
    dummy_audio = np.random.randn(int(SAMPLE_RATE * 3))
    features = extractor.extract(dummy_audio)
    print(f"  Spectrogram feature dim: {features.shape}")

    print("[AudioBranch] Wav2Vec2 test...")
    wav2vec_ext = AudioFeatureExtractor()
    features = wav2vec_ext.extract_features(dummy_audio)
    print(f"  Wav2Vec2 feature dim: {features.shape}")
    print("[AudioBranch] ✅ Self-test passed.")
