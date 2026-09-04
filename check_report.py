import sys
sys.path.insert(0, "src")
import pandas as pd

df = pd.read_csv("data/processed/structural_features.csv")

report = df.groupby("protein_id").agg(
    n_mutations=("position", "count"),
    n_valid_rsa=("rsa", lambda s: s.notna().sum()),
    n_nan_rsa=("rsa", lambda s: s.isna().sum()),
).reset_index()
report["status"] = report["n_nan_rsa"].apply(lambda n: "FULL FAIL" if n == report["n_mutations"].max() else ("PARTIAL" if n > 0 else "OK"))
report = report.sort_values("n_nan_rsa", ascending=False)

report.to_csv("data/processed/structural_features_report.csv", index=False)
print(report.to_string())
print()
print("Proteins fully failed:", (report["n_nan_rsa"] == report["n_mutations"]).sum())
print("Proteins partially failed:", ((report["n_nan_rsa"] > 0) & (report["n_nan_rsa"] < report["n_mutations"])).sum())
print("Proteins fully OK:", (report["n_nan_rsa"] == 0).sum())
