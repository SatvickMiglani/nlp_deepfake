"""
Fusion Models and Ablation Runner
Combines branch scores under all 7 ablation configurations.
Fusion strategies: mean score, logistic regression, shallow MLP.
"""
import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    DEVICE, ABLATION_CONFIGS, SCORE_CACHE_DIR, RESULTS_DIR,
    LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS_FUSION, EARLY_STOP_PATIENCE,
    RANDOM_SEED, TEMPORAL_AGG_STRATEGIES, TOP_K
)


# ─── Temporal Aggregation ────────────────────────────────────────────────────

def aggregate_segments(segment_scores, strategy="mean", k=TOP_K):
    """
    Aggregate per-segment scores into a single clip-level score.
    """
    if segment_scores is None or len(segment_scores) == 0:
        return 0.5  # Uncertain

    if strategy == "mean":
        return float(np.mean(segment_scores))
    elif strategy == "max":
        return float(np.max(segment_scores))
    elif strategy == "top_k":
        k = min(k, len(segment_scores))
        top_k = np.sort(segment_scores)[-k:]
        return float(np.mean(top_k))
    else:
        return float(np.mean(segment_scores))


# ─── Fusion Models ────────────────────────────────────────────────────────────

class MeanFusion:
    """Simple averaging of branch scores."""

    def __init__(self):
        self.name = "mean"

    def fit(self, X, y):
        pass  # No training needed

    def predict_proba(self, X):
        """Return averaged scores."""
        return np.mean(X, axis=1)

    def predict(self, X, threshold=0.5):
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)


class LogisticFusion:
    """Logistic regression on branch scores."""

    def __init__(self):
        self.name = "logistic"
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=1.0, max_iter=1000, random_state=RANDOM_SEED, class_weight='balanced'
        )

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X, threshold=0.5):
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def get_weights(self):
        """Return learned weights for interpretability."""
        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        }


class MLPFusion(nn.Module):
    """Shallow MLP for score fusion."""

    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.name = "mlp"
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPFusionWrapper:
    """Wrapper to train and use the MLP fusion model."""

    def __init__(self, input_dim, device=DEVICE):
        self.name = "mlp"
        self.device = device
        self.model = MLPFusion(input_dim).to(device)
        self.scaler = StandardScaler()

    def fit(self, X, y, X_val=None, y_val=None):
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCELoss()

        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(NUM_EPOCHS_FUSION):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            optimizer.step()

            # Early stopping on validation
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    X_val_scaled = self.scaler.transform(X_val)
                    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
                    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    val_preds = self.model(X_val_t)
                    val_loss = criterion(val_preds, y_val_t).item()
                self.model.train()

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOP_PATIENCE:
                        print(f"  [MLP] Early stopping at epoch {epoch+1}")
                        break

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
            scores = self.model(X_tensor).cpu().numpy()
        return scores

    def predict(self, X, threshold=0.5):
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)


# ─── Score Cache Manager ─────────────────────────────────────────────────────

class ScoreCacheManager:
    """Save and load per-sample, per-branch cached scores."""

    def __init__(self, cache_dir=SCORE_CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def save_scores(self, sample_id, branch_name, segment_scores):
        """Save segment scores for a sample."""
        path = os.path.join(self.cache_dir, f"{sample_id}_{branch_name}.npy")
        np.save(path, segment_scores)

    def load_scores(self, sample_id, branch_name):
        """Load segment scores for a sample."""
        path = os.path.join(self.cache_dir, f"{sample_id}_{branch_name}.npy")
        if os.path.exists(path):
            return np.load(path)
        return None

    def has_scores(self, sample_id, branch_name):
        path = os.path.join(self.cache_dir, f"{sample_id}_{branch_name}.npy")
        return os.path.exists(path)

    def save_features(self, sample_id, branch_name, features):
        """Save raw features for a sample."""
        path = os.path.join(self.cache_dir, f"{sample_id}_{branch_name}_features.npy")
        np.save(path, features)

    def load_features(self, sample_id, branch_name):
        """Load raw features for a sample."""
        path = os.path.join(self.cache_dir, f"{sample_id}_{branch_name}_features.npy")
        if os.path.exists(path):
            return np.load(path)
        return None


# ─── Ablation Evaluator ──────────────────────────────────────────────────────

class AblationEvaluator:
    """Run all 7 ablation configurations and collect metrics."""

    def __init__(self, cache_manager=None):
        self.cache = cache_manager or ScoreCacheManager()
        self.results = {}

    def build_feature_matrix(self, sample_ids, config_name, agg_strategy="mean"):
        """
        Build the feature matrix for a given ablation config.
        Columns correspond to active branches.
        """
        config = ABLATION_CONFIGS[config_name]
        active_branches = [b for b, active in config.items() if active]

        X = []
        valid_indices = []
        for i, sid in enumerate(sample_ids):
            row = []
            valid = True
            for branch in active_branches:
                scores = self.cache.load_scores(sid, branch)
                if scores is None:
                    valid = False
                    break
                clip_score = aggregate_segments(scores, strategy=agg_strategy)
                row.append(clip_score)
            if valid:
                X.append(row)
                valid_indices.append(i)

        return np.array(X) if X else None, valid_indices

    def evaluate_config(self, config_name, train_ids, train_labels, test_ids, test_labels,
                        val_ids=None, val_labels=None, agg_strategy="mean"):
        """
        Evaluate a single ablation config with all fusion strategies.
        Returns dict of {fusion_name: metrics_dict}.
        """
        print(f"\n{'='*60}")
        print(f"Evaluating config: {config_name} (aggregation: {agg_strategy})")
        print(f"{'='*60}")

        # Build matrices
        X_train, train_valid = self.build_feature_matrix(train_ids, config_name, agg_strategy)
        X_test, test_valid = self.build_feature_matrix(test_ids, config_name, agg_strategy)

        if X_train is None or X_test is None:
            print(f"  ❌ Insufficient cached scores for {config_name}")
            return None

        y_train = np.array([train_labels[i] for i in train_valid])
        y_test = np.array([test_labels[i] for i in test_valid])

        X_val, y_val = None, None
        if val_ids is not None:
            X_val, val_valid = self.build_feature_matrix(val_ids, config_name, agg_strategy)
            if X_val is not None:
                y_val = np.array([val_labels[i] for i in val_valid])

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Feature columns: {[b for b, a in ABLATION_CONFIGS[config_name].items() if a]}")

        results = {}
        n_features = X_train.shape[1]

        # 1. Mean fusion
        mean_fuser = MeanFusion()
        mean_preds = mean_fuser.predict(X_test)
        mean_probs = mean_fuser.predict_proba(X_test)
        results["mean"] = compute_metrics(y_test, mean_preds, mean_probs)
        print(f"  Mean fusion: {results['mean']}")

        # 2. Logistic regression
        lr_fuser = LogisticFusion()
        lr_fuser.fit(X_train, y_train)
        lr_preds = lr_fuser.predict(X_test)
        lr_probs = lr_fuser.predict_proba(X_test)
        results["logistic"] = compute_metrics(y_test, lr_preds, lr_probs)
        results["logistic"]["weights"] = lr_fuser.get_weights()
        print(f"  Logistic fusion: {results['logistic']}")

        # 3. MLP fusion
        mlp_fuser = MLPFusionWrapper(n_features)
        mlp_fuser.fit(X_train, y_train, X_val, y_val)
        mlp_preds = mlp_fuser.predict(X_test)
        mlp_probs = mlp_fuser.predict_proba(X_test)
        results["mlp"] = compute_metrics(y_test, mlp_preds, mlp_probs)
        print(f"  MLP fusion: {results['mlp']}")

        return results

    def run_all_configs(self, train_ids, train_labels, test_ids, test_labels,
                        val_ids=None, val_labels=None, agg_strategy="mean"):
        """Run all 7 ablation configs and return comprehensive results."""
        all_results = {}
        for config_name in ABLATION_CONFIGS:
            result = self.evaluate_config(
                config_name, train_ids, train_labels, test_ids, test_labels,
                val_ids, val_labels, agg_strategy
            )
            if result is not None:
                all_results[config_name] = result

        self.results = all_results
        return all_results

    def run_temporal_comparison(self, train_ids, train_labels, test_ids, test_labels,
                                 val_ids=None, val_labels=None):
        """Compare temporal aggregation strategies for the full A+V+E config."""
        temporal_results = {}
        for strategy in TEMPORAL_AGG_STRATEGIES:
            result = self.evaluate_config(
                "A+V+E", train_ids, train_labels, test_ids, test_labels,
                val_ids, val_labels, agg_strategy=strategy
            )
            if result is not None:
                temporal_results[strategy] = result

        return temporal_results

    def save_results(self, path=None):
        """Save results to JSON."""
        if path is None:
            path = os.path.join(RESULTS_DIR, "ablation_results.json")
        # Convert numpy types for JSON serialization
        results_clean = json.loads(json.dumps(self.results, default=str))
        with open(path, "w") as f:
            json.dump(results_clean, f, indent=2)
        print(f"[Evaluator] Results saved to {path}")


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute all evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = None
    return metrics


def print_comparison_table(results, fusion_method="logistic"):
    """Print a clean comparison table of all ablation configs."""
    print("\n" + "=" * 80)
    print(f"ABLATION COMPARISON TABLE (fusion: {fusion_method})")
    print("=" * 80)
    print(f"{'Config':<10} {'Accuracy':<10} {'Macro-F1':<10} {'ROC-AUC':<10} {'Precision':<11} {'Recall':<10}")
    print("-" * 80)
    for config_name in ABLATION_CONFIGS:
        if config_name in results and fusion_method in results[config_name]:
            m = results[config_name][fusion_method]
            auc = f"{m.get('roc_auc', 'N/A'):.4f}" if m.get('roc_auc') else "N/A"
            print(f"{config_name:<10} {m['accuracy']:<10.4f} {m['macro_f1']:<10.4f} "
                  f"{auc:<10} {m['precision']:<11.4f} {m['recall']:<10.4f}")
    print("=" * 80)


if __name__ == "__main__":
    print("[Fusion] Module loaded successfully. Ready for ablation experiments.")
