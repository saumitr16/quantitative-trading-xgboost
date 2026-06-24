import pandas as pd

df = pd.read_csv("data/processed/master_dataset.csv")

print(df.shape)
print(df.columns)
print(df.head())
print(df["ticker"].nunique())