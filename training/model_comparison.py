# ============================================================
# MODEL COMPARISON & EMPIRICAL BENCHMARK SCRIPT
# Breast Cancer XAI Framework (BreakHis 400X)
# ============================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.config import RESULTS_DIR


def generate_comparison_report():
    """
    Consolidates evaluation metrics across CNN, ResNet50, and EfficientNetB0,
    generates tabular summary and publication-ready comparative bar plot.
    """
    output_dir = os.path.join(RESULTS_DIR, "model_comparison")
    os.makedirs(output_dir, exist_ok=True)

    metrics = [
        {
            "Model": "CNN",
            "Accuracy": 80.49,
            "Precision": 84.67,
            "Recall": 80.49,
            "F1 Score": 81.03,
            "Macro AUC": 97.95
        },
        {
            "Model": "ResNet50",
            "Accuracy": 84.41,
            "Precision": 84.76,
            "Recall": 84.41,
            "F1 Score": 83.44,
            "Macro AUC": 98.35
        },
        {
            "Model": "EfficientNetB0",
            "Accuracy": 60.14,
            "Precision": 62.82,
            "Recall": 60.14,
            "F1 Score": 52.17,
            "Macro AUC": 89.14
        }
    ]

    df = pd.DataFrame(metrics)
    csv_path = os.path.join(output_dir, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Model comparison CSV written to: {csv_path}")

    # Plot
    df_melted = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    palette = ["#4361ee", "#2ec4b6", "#e63946", "#f77f00", "#7209b7"]

    ax = sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", palette="viridis")
    plt.title("Comparative Performance Analysis (BreakHis 400X)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Evaluation Metric", fontsize=12, labelpad=10)
    plt.ylabel("Performance Score (%)", fontsize=12, labelpad=10)
    plt.ylim(0, 110)
    plt.legend(title="Model Architecture", loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3),
                        textcoords='offset points')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[OK] Model comparison figure saved to: {plot_path}")


if __name__ == "__main__":
    generate_comparison_report()
