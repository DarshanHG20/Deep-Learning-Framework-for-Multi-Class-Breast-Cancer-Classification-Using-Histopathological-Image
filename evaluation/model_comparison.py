import matplotlib.pyplot as plt
import pandas as pd
import os

# ==============================
# MODEL PERFORMANCE DATA
# ==============================

data = {
    "Model": [
        "CNN",
        "ResNet50",
        "EfficientNetB0"
    ],
    "Accuracy": [
        80.49,
        84.41,
        60.14
    ],
    "Precision": [
        84.67,
        84.76,
        62.82
    ],
    "Recall": [
        80.49,
        84.41,
        60.14
    ],
    "F1 Score": [
        81.03,
        83.44,
        52.17
    ],
    "Macro AUC": [
        97.95,
        98.35,
        89.14
    ]
}

df = pd.DataFrame(data)

# ==============================
# CREATE RESULTS DIRECTORY
# ==============================

results_dir = "results/model_comparison"
os.makedirs(results_dir, exist_ok=True)

# ==============================
# SAVE CSV
# ==============================

df.to_csv(
    os.path.join(results_dir, "model_comparison.csv"),
    index=False
)

# ==============================
# PRINT TABLE
# ==============================

print("\n==============================================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================================\n")

print(df.to_string(index=False))

# ==============================
# PLOT
# ==============================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Macro AUC"
]

x = range(len(df["Model"]))
width = 0.15

plt.figure(figsize=(12, 7))

for i, metric in enumerate(metrics):
    values = df[metric].values

    positions = [
        pos + (i - 2) * width
        for pos in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"]
)

plt.ylabel("Performance (%)")
plt.xlabel("Deep Learning Model")
plt.title("Performance Comparison of Deep Learning Models")

plt.ylim(0, 105)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.4
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        results_dir,
        "model_comparison.png"
    ),
    dpi=300
)

plt.show()

print("\n==============================================")
print("MODEL COMPARISON COMPLETED")
print("==============================================")

print("\nSaved files:")
print("results/model_comparison/model_comparison.csv")
print("results/model_comparison/model_comparison.png")