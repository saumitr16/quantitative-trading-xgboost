import yfinance as yf

df = yf.download(
    "^NSEI",
    start="2015-01-01",
    end="2025-01-01",
    auto_adjust=True
)

if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
    df.columns = df.columns.get_level_values(0)

df.reset_index(inplace=True)

df.to_csv(
    "data/raw/NIFTY.csv",
    index=False
)

print(df.head())