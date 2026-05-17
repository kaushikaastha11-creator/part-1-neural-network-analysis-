# Part 1 – Neural Network Fundamentals and Training Behavior Analysis

## Project Overview

This project builds and analyses a **feed-forward neural network** for binary customer churn prediction using the `customer_churn_nn.csv` dataset (2,000 records, 15 features, 1 target).

---

## Dataset

| Property | Value |
|---|---|
| File | `customer_churn_nn.csv` |
| Rows | 2,000 |
| Features | 15 (4 categorical, 11 numerical) |
| Target | `churn` — 0 = Retained, 1 = Churned |
| Churn Rate | ~1.6% (severely imbalanced) |

**Categorical features:** `region`, `plan_type`, `contract_type`, `payment_method`  
**Numerical features:** `tenure_months`, `monthly_charges_inr`, `avg_login_days_per_month`, `support_tickets_last_90_days`, `payment_delay_days`, `data_usage_gb`, `satisfaction_score`, `last_complaint_days_ago`, `discount_percent`, `autopay_enabled`, `referral_count`

---

## Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb               ← Full analysis notebook (Tasks 1–6)
├── run_analysis.py              ← Standalone Python script
├── customer_churn_nn.csv        ← Dataset
├── requirements.txt             ← Python dependencies
└── results/
    ├── task1_eda.png            ← EDA visualisations (Task 1)
    ├── task4_evaluation_outputs.png  ← Loss/Accuracy curves + Confusion Matrix
    ├── task5_model_comparison.png   ← Hyperparameter comparison charts
    ├── task5_training_curves.png    ← Training curves for all experiments
    ├── model_comparison_table.csv   ← Experiment results table
    └── model_comparison_table.png   ← Experiment results table (image)
```

---

## Tasks Summary

### Task 1 – Dataset Understanding
- 2,000 rows, 17 columns (including `customer_id`)
- Zero missing values across all features
- Target heavily imbalanced: 1,969 retained vs 31 churned (1.6% churn rate)
- Key observations: higher support tickets and payment delays correlate with churn

### Task 2 – Data Preprocessing
- Dropped `customer_id` (identifier, not predictive)
- Label-encoded 4 categorical columns
- Applied `StandardScaler` to all 15 features
- Stratified 80/20 train-test split (1,600 train / 400 test)

### Task 3 – Neural Network Architecture (Baseline)
```
Input Layer  → 15 neurons (one per feature)
Hidden Layer → 64 neurons, ReLU activation, Dropout(0.2)
Output Layer → 1 neuron, Sigmoid activation
Loss         → Binary Cross-Entropy
Optimizer    → Adam (lr = 0.001)
```

### Task 4 – Training and Evaluation
| Metric | Value |
|---|---|
| Test Accuracy | 98.5% |
| AUC-ROC | ~0.87 |
| Churn Recall | ~0% (class imbalance issue) |

> Raw accuracy is misleading here. AUC-ROC is the primary metric given severe class imbalance.

### Task 5 – Hyperparameter Experiments

| Experiment | Hidden Layers | Neurons | LR | Batch | Activation | Accuracy (%) | AUC-ROC |
|---|---|---|---|---|---|---|---|
| Baseline | 1 | 64 | 0.001 | 32 | ReLU | 98.50 | 0.875 |
| Deeper (3L) | 3 | 64 | 0.001 | 32 | ReLU | 98.50 | 0.828 |
| Wider (128N) | 1 | 128 | 0.001 | 32 | ReLU | 98.75 | 0.845 |
| High LR | 1 | 64 | 0.01 | 32 | ReLU | 98.50 | 0.817 |
| Low LR | 1 | 64 | 0.0001 | 32 | ReLU | 98.50 | 0.854 |
| Large Batch | 1 | 64 | 0.001 | 128 | ReLU | 98.50 | 0.907 |
| Tanh | 1 | 64 | 0.001 | 32 | Tanh | 98.50 | **0.954** |

**Best AUC-ROC:** Tanh Activation (0.954) — smoother gradient flow helps with imbalanced data.

### Task 6 – Final Reflection

**Weights & Biases:** Weights control connection strength between neurons; biases shift activation thresholds. Both are updated via backpropagation to minimize loss.

**Activation Functions:** Introduce non-linearity, allowing the network to learn complex patterns. Without them, stacked layers collapse to a single linear transformation.

**Learning Rate Effects:**
- Too high → loss diverges or oscillates
- Too low → very slow convergence, risk of local minima
- Well-tuned (0.001) → stable convergence

**Overfitting/Underfitting:** The model shows **underfitting on the minority class** due to class imbalance. It achieves high accuracy by predicting "Retained" for nearly all samples, missing almost all churners. No overfitting gap was observed between training and validation loss.

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run as Python script
python run_analysis.py

# Or open the Jupyter notebook
jupyter notebook notebook.ipynb
```

---

## Requirements

See `requirements.txt`. Core libraries: TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.
