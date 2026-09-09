import pandas as pd

df = pd.read_csv("data/processed/df_with_error_and_strata.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3).to_string())
print()
for col in df.columns:
    if df[col].dtype == object or df[col].nunique() < 15:
        print(col, "unique values:", df[col].dropna().unique()[:15])