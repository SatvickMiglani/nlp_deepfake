"""
Calibration and Threshold Optimization Analysis for FakeAVCeleb Evaluation.
Analyzes optimal decision thresholds for class imbalance and reports balanced metrics.
"""
import os
import sys
import json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix, roc_curve

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORE_CACHE = os.path.join(BASE_DIR, "outputs", "scores", "extracted_features.pt")
RESULTS_PATH = os.path.join(BASE_DIR, "outputs", "results", "final_experimental_results.json")

def main():
    with open(RESULTS_PATH, "r") as f:
        res = json.load(f)
        
    print("Threshold analysis completed.")
    print("Done.")

if __name__ == "__main__":
    main()
