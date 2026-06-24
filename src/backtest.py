import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/processed/predictions.csv"
)

df["Date"] = pd.to_datetime(df["Date"])

dates = sorted(df["Date"].unique())

portfolio_returns = []

for date in dates:

    day_df = df[df["Date"] == date]

    top5 = (
        day_df
        .sort_values(
            "predicted_return",
            ascending=False
        )
        .head(5)
    )

    portfolio_return = top5["target_return"].mean()

    portfolio_returns.append(
        portfolio_return
    )

returns = pd.Series(
    portfolio_returns,
    index=dates
)

# ==================================
# Equity Curve
# ==================================

equity_curve = (
    1 + returns
).cumprod()

initial_capital = 100000

portfolio_value = (
    initial_capital
    * equity_curve
)

# ==================================
# CAGR
# ==================================

years = len(returns) / 252

cagr = (
    portfolio_value.iloc[-1]
    /
    portfolio_value.iloc[0]
) ** (1 / years) - 1

# ==================================
# Sharpe
# ==================================

sharpe = (
    returns.mean()
    /
    returns.std()
) * np.sqrt(252)

# ==================================
# Drawdown
# ==================================

rolling_max = equity_curve.cummax()

drawdown = (
    equity_curve
    /
    rolling_max
    - 1
)

max_drawdown = drawdown.min()

# ==================================
# Win Rate
# ==================================

win_rate = (
    returns > 0
).mean()

# ==================================
# Results
# ==================================

print("=" * 50)
print("Backtest Results")
print("=" * 50)

print(
    f"Final Portfolio Value : ₹{portfolio_value.iloc[-1]:,.2f}"
)

print(
    f"CAGR                 : {cagr:.2%}"
)

print(
    f"Sharpe Ratio         : {sharpe:.2f}"
)

print(
    f"Max Drawdown         : {max_drawdown:.2%}"
)

print(
    f"Win Rate             : {win_rate:.2%}"
)
