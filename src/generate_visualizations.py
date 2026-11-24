"""
Generate publication-quality figures and plots for the Multimodal Deepfake Detection research study.
Loads final_experimental_results.json and creates clean visualizations.
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_JSON = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "results", "final_experimental_results.json")

def set_plot_style():
    plt.style.use("default")
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7

def plot_ablation_comparison(data):
    ablations = data["ablation_results"]
    configs = ["A", "V", "E", "A+V", "A+E", "V+E", "A+V+E"]
    
    accuracies = [ablations[c]["logistic"]["accuracy"] * 100 for c in configs]
    f1_scores = [ablations[c]["logistic"]["macro_f1"] * 100 for c in configs]
    roc_aucs = [ablations[c]["logistic"]["roc_auc"] * 100 for c in configs]
    
    x = np.arange(len(configs))
    width = 0.26
    
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    
    rects1 = ax.bar(x - width, accuracies, width, label="Accuracy (%)", color="#2b5c8f", alpha=0.9)
    rects2 = ax.bar(x, f1_scores, width, label="Macro-F1 (%)", color="#388e3c", alpha=0.9)
    rects3 = ax.bar(x + width, roc_aucs, width, label="ROC-AUC (%)", color="#e65100", alpha=0.9)
    
    ax.set_ylabel("Metric Value (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance Across All 7 Ablation Configurations (Logistic Fusion)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10, loc="lower right")
    
    # Add value annotations on top of ROC-AUC bars
    for rect in rects3:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
                    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig1_ablation_study.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_manipulation_stratification(data):
    strat = data["stratified_results"]
    cats = ["RealVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-RealAudio", "FakeVideo-FakeAudio"]
    labels = ["Real / Real\n(Authentic)", "RealV / FakeA\n(Voice Clone)", "FakeV / RealA\n(Face Swap)", "FakeV / FakeA\n(Dual Fake)"]
    
    accs = [strat[c]["accuracy"] * 100 for c in cats]
    counts = [strat[c]["count"] for c in cats]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    colors = ["#457b9d", "#e76f51", "#f4a261", "#2a9d8f"]
    bars = ax.bar(labels, accs, color=colors, width=0.55, edgecolor="#333333", linewidth=1.0)
    
    ax.set_ylabel("Detection Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Detection Accuracy Stratified by Manipulation Type (RQ3)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y")
    
    for bar, count in zip(bars, counts):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%\n(N={count})",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig2_manipulation_stratification.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_fusion_comparison(data):
    ablations = data["ablation_results"]["A+V+E"]
    fusions = ["mean", "logistic", "mlp"]
    labels = ["Mean Averaging\n(Equal Voting)", "Logistic Regression\n(Learned Weights)", "Shallow MLP\n(Nonlinear Fusion)"]
    
    accs = [ablations[f]["accuracy"] * 100 for f in fusions]
    f1s = [ablations[f]["macro_f1"] * 100 for f in fusions]
    aucs = [ablations[f]["roc_auc"] * 100 for f in fusions]
    
    x = np.arange(len(fusions))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(x - width, accs, width, label="Accuracy (%)", color="#1d3557")
    ax.bar(x, f1s, width, label="Macro-F1 (%)", color="#457b9d")
    ax.bar(x + width, aucs, width, label="ROC-AUC (%)", color="#e63946")
    
    ax.set_ylabel("Metric Value (%)", fontsize=12, fontweight="bold")
    ax.set_title("Comparison of Fusion Architectures for Tripartite Model (A+V+E)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(frameon=True, loc="lower right")
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig3_fusion_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_hard_cases(data):
    hard_cases = data["top_15_hard_cases"][:8] # Top 8 most divergent
    vids = [c["video_id"].split("__")[1][:12] for c in hard_cases]
    p_a = [c["prob_audio"] for c in hard_cases]
    p_v = [c["prob_visual"] for c in hard_cases]
    p_e = [c["prob_emotion"] for c in hard_cases]
    
    x = np.arange(len(vids))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(x - width, p_a, width, label="P(Audio Fake)", color="#2a9d8f")
    ax.bar(x, p_v, width, label="P(Visual Fake)", color="#e76f51")
    ax.bar(x + width, p_e, width, label="P(Emotion Dissonance)", color="#f4a261")
    
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.6, label="Decision Threshold (0.5)")
    ax.set_ylabel("Predicted Fake Probability", fontsize=11, fontweight="bold")
    ax.set_title("Modality Prediction Disagreement Across Hard Error Cases", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(vids, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y")
    ax.legend(frameon=True, loc="upper right", fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "fig4_hard_cases_disagreement.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def main():
    set_plot_style()
    with open(RESULTS_JSON, "r") as f:
        data = json.load(f)
        
    plot_ablation_comparison(data)
    plot_manipulation_stratification(data)
    plot_fusion_comparison(data)
    plot_hard_cases(data)
    print("\nAll figures successfully generated in outputs/figures/")

if __name__ == "__main__":
    main()
