import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ====================================
# Load Dataset
# ====================================

df = pd.read_csv(
    "data/processed/master_dataset.csv"
)

print("=" * 50)
print("Dataset")
print("=" * 50)

print(df.shape)
print()

# ====================================
# Convert Date
# ====================================

df["Date"] = pd.to_datetime(df["Date"])

# ====================================
# Features
# ====================================

features = [
    "sma_10",
    "sma_20",
    "sma_50",
    "rsi",
    "macd",
    "macd_signal",
    "bb_upper",
    "bb_lower",
    "return_1d",
    "return_5d",
    "volume_change",
    "nifty_return_1d",
    "nifty_return_5d",
    "volatility_5",
    "volatility_20",
    "close_sma20_ratio",
    "close_sma50_ratio",
    "sma10_sma50_ratio"
]

target = "target_class"

X = df[features]
y = df[target]

# ====================================
# Time Split
# ====================================

split_date = df["Date"].quantile(0.80)

train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print(f"Train Samples : {len(X_train)}")
print(f"Test Samples  : {len(X_test)}")
print()

# ====================================
# Class Balance
# ====================================

print("Target Distribution")
print("-" * 30)

print(y.value_counts(normalize=True))

print()

# ====================================
# Model
# ====================================

model = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric="logloss"
)

print("Training...")

model.fit(
    X_train,
    y_train
)

print("Done.\n")

# ====================================
# Predictions
# ====================================

preds = model.predict(X_test)

probs = model.predict_proba(X_test)[:, 1]

# ====================================
# Metrics
# ====================================

accuracy = accuracy_score(
    y_test,
    preds
)

precision = precision_score(
    y_test,
    preds,
    zero_division=0
)

recall = recall_score(
    y_test,
    preds,
    zero_division=0
)

f1 = f1_score(
    y_test,
    preds,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probs
)

print("=" * 50)
print("Classification Metrics")
print("=" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")

print()

# ====================================
# Confusion Matrix
# ====================================

cm = confusion_matrix(
    y_test,
    preds
)

print("=" * 50)
print("Confusion Matrix")
print("=" * 50)

print(cm)

print()

# ====================================
# Feature Importance
# ====================================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("=" * 50)
print("Feature Importance")
print("=" * 50)

print(importance_df)

print()

# ====================================
# Top Probability Predictions
# ====================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Probability_Up": probs,
    "Prediction": preds
})

print("=" * 50)
print("Sample Predictions")
print("=" * 50)

print(
    results.sort_values(
        "Probability_Up",
        ascending=False
    ).head(20)
)