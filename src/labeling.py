import pandas as pd

def create_target(df):

    future_avg = (
        df["Close"]
        .shift(-5)
        .rolling(5)
        .mean()
    )

    df["target_return"] = (
        future_avg / df["Close"] - 1
    )
    df["target_class"] = (
    df["target_return"] > 0.02
    ).astype(int)
    return df