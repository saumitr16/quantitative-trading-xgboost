import pandas as pd

from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


def add_features(df):

    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["sma_10"] = SMAIndicator(close, window=10).sma_indicator()
    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close, window=50).sma_indicator()

    # RSI
    df["rsi"] = RSIIndicator(close, window=14).rsi()

    # MACD
    macd = MACD(close)

    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(close)

    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()

    # Returns
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)

    # Volume
    df["volume_change"] = volume.pct_change()

    # Volatility

    df["volatility_5"] = (
        df["return_1d"]
        .rolling(5)
        .std()
    )

    df["volatility_20"] = (
        df["return_1d"]
        .rolling(20)
        .std()
    )

    # Relative position

    df["close_sma20_ratio"] = (
        df["Close"] / df["sma_20"]
    )

    df["close_sma50_ratio"] = (
        df["Close"] / df["sma_50"]
    )

    # Trend strength

    df["sma10_sma50_ratio"] = (
        df["sma_10"] / df["sma_50"]
    )

    return df