import pandas as pd
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/processed/RELIANCE.csv")

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
    "volume_change"
]

target = "target_return"

X = df[features]
y = df[target]


# =========================
# Time-Series Train/Test Split
# =========================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Train Samples : {len(X_train)}")
print(f"Test Samples  : {len(X_test)}")

print()


# =========================
# Model
# =========================

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training model...\n")

model.fit(X_train, y_train)

print("Training completed.\n")


# =========================
# Predictions
# =========================

preds = model.predict(X_test)


# =========================
# Regression Metrics
# =========================

mae = mean_absolute_error(y_test, preds)

rmse = mean_squared_error(
    y_test,
    preds
) ** 0.5

r2 = r2_score(
    y_test,
    preds
)

print("=" * 50)
print("Regression Metrics")
print("=" * 50)

print(f"MAE  : {mae:.6f}")
print(f"RMSE : {rmse:.6f}")
print(f"R²   : {r2:.6f}")

print()


# =========================
# Directional Accuracy
# =========================

actual_direction = (
    y_test > 0
).astype(int)

pred_direction = (
    preds > 0
).astype(int)

directional_accuracy = (
    actual_direction == pred_direction
).mean()

print("=" * 50)
print("Trading Metrics")
print("=" * 50)

print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.4f}"
)

print()


# =========================
# Feature Importance
# =========================

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)

print("=" * 50)
print("Feature Importance")
print("=" * 50)

print(importance_df)

print()


# =========================
# Sample Predictions
# =========================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds
})

print("=" * 50)
print("Sample Predictions")
print("=" * 50)

print(results.head(10))