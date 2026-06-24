from pathlib import Path
import pandas as pd

from features import add_features
from labeling import create_target

raw_dir = Path("data/raw")

all_data = []
nifty = pd.read_csv(
    "data/processed/nifty_features.csv"
)
for file in raw_dir.glob("*.csv"):

    print(f"Processing {file.name}")

    df = pd.read_csv(file)

    if "index" in df.columns:
        df = df.drop(columns=["index"])

    df = add_features(df)

    df = create_target(df)
    df = pd.merge(
        df,
        nifty,
        on="Date",
        how="left"
    )

    df["ticker"] = file.stem
    
    import numpy as np

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )
    df.dropna(inplace=True)

    all_data.append(df)

master_df = pd.concat(
    all_data,
    ignore_index=True
)

print(master_df.shape)

master_df.to_csv(
    "data/processed/master_dataset.csv",
    index=False
)