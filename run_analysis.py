"""
Part 1: Neural Network Fundamentals and Training Behavior Analysis
Dataset: customer_churn_nn.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, roc_auc_score)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)
np.random.seed(42)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# TASK 1: DATASET UNDERSTANDING
# =============================================================================
print("=" * 65)
print("TASK 1: DATASET UNDERSTANDING")
print("=" * 65)

df = pd.read_csv("customer_churn_nn.csv")
print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nStatistical Summary:\n{df.describe().T.round(2)}")
print(f"\nTarget Distribution:\n{df['churn'].value_counts()}")
print(f"Churn Rate: {df['churn'].mean()*100:.1f}%")

# ── EDA figure ───────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("Task 1 – Dataset Exploration", fontsize=16, fontweight="bold", y=1.01)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# (a) Churn distribution
ax0 = fig.add_subplot(gs[0, 0])
counts = df["churn"].value_counts()
bars = ax0.bar(["Retained (0)", "Churned (1)"], counts.values,
               color=["#4CAF50", "#F44336"], edgecolor="white", linewidth=1.2)
for bar, val in zip(bars, counts.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
             str(val), ha="center", fontsize=11, fontweight="bold")
ax0.set_title("Target Distribution", fontweight="bold")
ax0.set_ylabel("Count")

# (b) Tenure histogram by churn
ax1 = fig.add_subplot(gs[0, 1])
for label, color in zip([0, 1], ["#4CAF50", "#F44336"]):
    ax1.hist(df[df["churn"] == label]["tenure_months"],
             bins=20, alpha=0.65, color=color,
             label=f"Churn={label}", edgecolor="white")
ax1.set_title("Tenure Distribution by Churn", fontweight="bold")
ax1.set_xlabel("Tenure (months)")
ax1.legend()

# (c) Monthly charges by churn
ax2 = fig.add_subplot(gs[0, 2])
df.boxplot(column="monthly_charges_inr", by="churn", ax=ax2,
           patch_artist=True,
           boxprops=dict(facecolor="#90CAF9"),
           medianprops=dict(color="red", linewidth=2))
ax2.set_title("Monthly Charges by Churn", fontweight="bold")
ax2.set_xlabel("Churn")
ax2.set_ylabel("Monthly Charges (INR)")
plt.sca(ax2); plt.title("Monthly Charges by Churn"); plt.suptitle("")

# (d) Satisfaction score histogram
ax3 = fig.add_subplot(gs[1, 0])
ax3.hist(df["satisfaction_score"], bins=15, color="#5C6BC0", edgecolor="white")
ax3.set_title("Satisfaction Score Distribution", fontweight="bold")
ax3.set_xlabel("Satisfaction Score")

# (e) Contract type vs churn
ax4 = fig.add_subplot(gs[1, 1])
ct = df.groupby(["contract_type", "churn"]).size().unstack(fill_value=0)
ct.plot(kind="bar", ax=ax4, color=["#4CAF50", "#F44336"],
        edgecolor="white", legend=True)
ax4.set_title("Contract Type vs Churn", fontweight="bold")
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=25, ha="right")
ax4.legend(["Retained", "Churned"])

# (f) Correlation heat-map (numeric only)
ax5 = fig.add_subplot(gs[1, 2])
num_cols = df.select_dtypes(include=np.number).drop(columns=["customer_id"], errors="ignore")
corr = num_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax5, cmap="coolwarm", center=0,
            linewidths=0.4, annot=False, cbar_kws={"shrink": 0.7})
ax5.set_title("Feature Correlation Matrix", fontweight="bold")
ax5.tick_params(labelsize=7)

# (g) Support tickets histogram
ax6 = fig.add_subplot(gs[2, 0])
ax6.hist(df["support_tickets_last_90_days"], bins=10,
         color="#FF8A65", edgecolor="white")
ax6.set_title("Support Tickets (90 days)", fontweight="bold")
ax6.set_xlabel("Number of Tickets")

# (h) Data usage by churn
ax7 = fig.add_subplot(gs[2, 1])
for label, color in zip([0, 1], ["#4CAF50", "#F44336"]):
    ax7.hist(df[df["churn"] == label]["data_usage_gb"],
             bins=20, alpha=0.65, color=color,
             label=f"Churn={label}", edgecolor="white")
ax7.set_title("Data Usage (GB) by Churn", fontweight="bold")
ax7.set_xlabel("Data Usage (GB)")
ax7.legend()

# (i) Payment delay histogram
ax8 = fig.add_subplot(gs[2, 2])
ax8.hist(df["payment_delay_days"], bins=15, color="#AB47BC", edgecolor="white")
ax8.set_title("Payment Delay (days)", fontweight="bold")
ax8.set_xlabel("Days")

fig.savefig(f"{RESULTS_DIR}/task1_eda.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n✓ Saved: results/task1_eda.png")

# =============================================================================
# TASK 2: DATA PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("TASK 2: DATA PREPROCESSING")
print("=" * 65)

df_proc = df.drop(columns=["customer_id"]).copy()

# No missing values – confirmed above
cat_cols = ["region", "plan_type", "contract_type", "payment_method"]
num_cols_list = [c for c in df_proc.columns
                 if c not in cat_cols + ["churn"]]

# Label-encode categorical features
le = LabelEncoder()
for col in cat_cols:
    df_proc[col] = le.fit_transform(df_proc[col])
    print(f"  Encoded '{col}'")

X = df_proc.drop(columns=["churn"]).values
y = df_proc["churn"].values

# Train / Test split (80/20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Standard scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\n  Training set : {X_train.shape}")
print(f"  Testing  set : {X_test.shape}")
print(f"  Features     : {X.shape[1]}")
print(f"  Churn % train: {y_train.mean()*100:.1f}%")
print(f"  Churn % test : {y_test.mean()*100:.1f}%")

INPUT_DIM = X_train.shape[1]

# =============================================================================
# TASK 3 & 4: MODEL BUILDING + TRAINING + EVALUATION (Baseline)
# =============================================================================
print("\n" + "=" * 65)
print("TASKS 3 & 4: MODEL BUILDING, TRAINING & EVALUATION")
print("=" * 65)

def build_model(hidden_layers=1, neurons=64, lr=0.001,
                activation="relu", dropout=0.2):
    model = Sequential()
    model.add(Dense(neurons, input_dim=INPUT_DIM, activation=activation))
    model.add(Dropout(dropout))
    for _ in range(hidden_layers - 1):
        model.add(Dense(neurons, activation=activation))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

def train_and_evaluate(model, epochs=50, batch_size=32, label=""):
    es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )
    loss, acc  = model.evaluate(X_test, y_test, verbose=0)
    y_pred_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_prob >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_pred_prob)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"\n  [{label}]")
    print(f"  Test Accuracy: {acc*100:.2f}%  |  AUC: {auc:.4f}")
    print(f"  Classification Report:\n{classification_report(y_test, y_pred, digits=3)}")
    return history, acc, auc, cm, y_pred_prob

# ── Baseline model ────────────────────────────────────────────────────────────
baseline = build_model(hidden_layers=1, neurons=64, lr=0.001,
                        activation="relu")
baseline.summary()
hist_base, acc_base, auc_base, cm_base, prob_base = train_and_evaluate(
    baseline, epochs=100, batch_size=32, label="Baseline")

# ── Evaluation figure (Baseline) ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Task 4 – Baseline Model Evaluation", fontsize=14, fontweight="bold")

# Loss curve
axes[0].plot(hist_base.history["loss"],     label="Train Loss",  color="#1565C0")
axes[0].plot(hist_base.history["val_loss"], label="Val Loss",   color="#EF5350", linestyle="--")
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend()

# Accuracy curve
axes[1].plot(hist_base.history["accuracy"],     label="Train Acc",  color="#2E7D32")
axes[1].plot(hist_base.history["val_accuracy"], label="Val Acc",   color="#FF7043", linestyle="--")
axes[1].set_title("Training vs Validation Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend()

# Confusion matrix
sns.heatmap(cm_base, annot=True, fmt="d", cmap="Blues", ax=axes[2],
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"])
axes[2].set_title(f"Confusion Matrix\nAcc={acc_base*100:.1f}%  AUC={auc_base:.3f}")
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")

plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/task4_evaluation_outputs.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\n✓ Saved: results/task4_evaluation_outputs.png")

# =============================================================================
# TASK 5: HYPERPARAMETER EXPERIMENTATION
# =============================================================================
print("\n" + "=" * 65)
print("TASK 5: HYPERPARAMETER EXPERIMENTATION")
print("=" * 65)

experiments = [
    # (label,          hidden, neurons, lr,    batch, epochs, activation)
    ("Baseline (1L-64-relu)",      1, 64,  0.001, 32,  100, "relu"),
    ("Deeper (3L-64-relu)",        3, 64,  0.001, 32,  100, "relu"),
    ("Wider (1L-128-relu)",        1, 128, 0.001, 32,  100, "relu"),
    ("High LR (1L-64-lr=0.01)",   1, 64,  0.01,  32,  100, "relu"),
    ("Low LR (1L-64-lr=0.0001)",  1, 64,  0.0001,32,  100, "relu"),
    ("Large Batch (batch=128)",    1, 64,  0.001, 128, 100, "relu"),
    ("Tanh Activation",            1, 64,  0.001, 32,  100, "tanh"),
]

results_rows = []
histories    = []

for (label, hl, neu, lr, bs, ep, act) in experiments:
    m = build_model(hidden_layers=hl, neurons=neu, lr=lr, activation=act)
    hist, acc, auc, cm, _ = train_and_evaluate(
        m, epochs=ep, batch_size=bs, label=label)
    histories.append((label, hist))
    results_rows.append({
        "Experiment":       label,
        "Hidden Layers":    hl,
        "Neurons":          neu,
        "Learning Rate":    lr,
        "Batch Size":       bs,
        "Activation":       act,
        "Test Accuracy (%)": round(acc * 100, 2),
        "AUC-ROC":          round(auc, 4),
    })

results_df = pd.DataFrame(results_rows)
results_df.to_csv(f"{RESULTS_DIR}/model_comparison_table.csv", index=False)
print("\n✓ Saved: results/model_comparison_table.csv")
print(results_df.to_string(index=False))

# ── Comparison figure ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Task 5 – Hyperparameter Experiment Comparison",
             fontsize=14, fontweight="bold")

colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
short_labels = [r["Experiment"].split(" ")[0] + "\n" + " ".join(r["Experiment"].split(" ")[1:])
                for r in results_rows]

# Accuracy bar chart
bars = axes[0].barh(range(len(results_rows)),
                    [r["Test Accuracy (%)"] for r in results_rows],
                    color=colors, edgecolor="white", height=0.65)
axes[0].set_yticks(range(len(results_rows)))
axes[0].set_yticklabels([r["Experiment"] for r in results_rows], fontsize=8)
axes[0].set_xlabel("Test Accuracy (%)")
axes[0].set_title("Test Accuracy by Configuration")
axes[0].set_xlim(50, 105)
for bar, row in zip(bars, results_rows):
    axes[0].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{row['Test Accuracy (%)']:.1f}%", va="center", fontsize=8)

# AUC bar chart
bars2 = axes[1].barh(range(len(results_rows)),
                     [r["AUC-ROC"] for r in results_rows],
                     color=colors, edgecolor="white", height=0.65)
axes[1].set_yticks(range(len(results_rows)))
axes[1].set_yticklabels([r["Experiment"] for r in results_rows], fontsize=8)
axes[1].set_xlabel("AUC-ROC")
axes[1].set_title("AUC-ROC by Configuration")
axes[1].set_xlim(0.5, 1.1)
for bar, row in zip(bars2, results_rows):
    axes[1].text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                 f"{row['AUC-ROC']:.3f}", va="center", fontsize=8)

plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/task5_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: results/task5_model_comparison.png")

# ── Loss curves for all experiments ──────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
fig.suptitle("Task 5 – Training Curves for All Experiments",
             fontsize=14, fontweight="bold")
axes = axes.flatten()
for i, (label, hist) in enumerate(histories):
    ax = axes[i]
    ax.plot(hist.history["loss"],     color="#1565C0", label="Train")
    ax.plot(hist.history["val_loss"], color="#EF5350", linestyle="--", label="Val")
    ax.set_title(label, fontsize=8, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=7)
    ax.set_ylabel("Loss", fontsize=7)
    ax.legend(fontsize=7)
if len(histories) < len(axes):
    axes[-1].set_visible(False)
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/task5_training_curves.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: results/task5_training_curves.png")

# ── Model comparison table as image ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 3.5))
ax.axis("off")
table_data = results_df.values.tolist()
col_labels = results_df.columns.tolist()
tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1565C0")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#E3F2FD")
fig.suptitle("Model Comparison Table", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{RESULTS_DIR}/model_comparison_table.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("✓ Saved: results/model_comparison_table.png")

print("\n" + "=" * 65)
print("ALL TASKS COMPLETE")
print("=" * 65)
