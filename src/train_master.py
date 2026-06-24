import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ====================================
# Load Master Dataset
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

    # New Features
    "volatility_5",
    "volatility_20",

    "close_sma20_ratio",
    "close_sma50_ratio",

    "sma10_sma50_ratio"
]

target = "target_return"

X = df[features]
y = df[target]

# ====================================
# Time Split
# ====================================

df["Date"] = pd.to_datetime(df["Date"])

split_date = df["Date"].quantile(0.80)

train_mask = df["Date"] < split_date
test_mask = df["Date"] >= split_date

X_train = X[train_mask]
X_test = X[test_mask]

y_train = y[train_mask]
y_test = y[test_mask]

print(f"Train Samples: {len(X_train)}")
print(f"Test Samples : {len(X_test)}")
print()

# ====================================
# Model
# ====================================

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

print("Training...")
import numpy as np

print("Checking infinities...\n")

for col in X.columns:

    inf_count = np.isinf(X[col]).sum()

    if inf_count > 0:
        print(f"{col}: {inf_count}")
model.fit(
    X_train,
    y_train
)

print("Done.\n")

# ====================================
# Predictions
# ====================================

preds = model.predict(X_test)
results_df = df.loc[test_mask].copy()

results_df["predicted_return"] = preds

results_df.to_csv(
    "data/processed/predictions.csv",
    index=False
)
# Average return of all stocks in test set
market_avg_return = results_df["target_return"].mean()

# Daily top-5 strategy
daily_top5_returns = []

for date, group in results_df.groupby("Date"):

    top5 = (
        group
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(5)
    )

    daily_top5_returns.append(
        top5["target_return"].mean()
    )

top5_avg_return = sum(daily_top5_returns) / len(daily_top5_returns)

# Win rate
win_rate = (
    pd.Series(daily_top5_returns) > 0
).mean()

print()
print("=" * 50)
print("Ranking Evaluation")
print("=" * 50)

print(
    f"Market Average Return : "
    f"{market_avg_return:.4%}"
)

print(
    f"Top-5 Average Return  : "
    f"{top5_avg_return:.4%}"
)

print(
    f"Top-5 Win Rate        : "
    f"{win_rate:.4%}"
)

print(
    "Market Average:",
    results_df["target_return"].mean()
)

print("Predictions saved.")
# ====================================
# Metrics
# ====================================

mae = mean_absolute_error(
    y_test,
    preds
)

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

# ====================================
# Directional Accuracy
# ====================================

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
# Sample Predictions
# ====================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds
})

print(results.head(20))