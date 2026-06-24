import yfinance as yf
import pandas as pd

def download_stock(symbol,
                   start="2015-01-01",
                   end="2025-01-01"):

    df = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True
    )
    # Flatten weird Yahoo columns if present
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df.reset_index(inplace=True)

    return df


if __name__ == "__main__":

    stocks = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "ITC.NS",
        "LT.NS",
        "BHARTIARTL.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "ASIANPAINT.NS",
        "MARUTI.NS",
        "BAJFINANCE.NS",
        "HCLTECH.NS",
        "SUNPHARMA.NS",
        "TITAN.NS",
        "WIPRO.NS",
        "ULTRACEMCO.NS",
        "NESTLEIND.NS"
    ]

    for stock in stocks:

        df = download_stock(stock)

        filename = stock.replace(".NS", "")

        df.to_csv(
            f"data/raw/{filename}.csv",
            index=False
        )

        print(f"Saved {stock}")