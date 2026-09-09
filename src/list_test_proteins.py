"""
Extract the unique protein_ids in the protein-level test split, so we can
run ThermoMPNN on exactly the same held-out proteins Person A used.

Reads:  data/processed/df_with_error_and_strata.csv
Writes: data/processed/test_protein_ids.txt  (one protein_id per line)
"""
import pandas as pd

df = pd.read_csv("data/processed/df_with_error_and_strata.csv", low_memory=False,
                  usecols=["protein_id", "split"])
test_proteins = sorted(df.loc[df["split"] == "test", "protein_id"].unique())
print(f"{len(test_proteins)} unique proteins in test split")

with open("data/processed/test_protein_ids.txt", "w") as f:
    for pid in test_proteins:
        f.write(pid + "\n")

print("Wrote data/processed/test_protein_ids.txt")
print(test_proteins[:10])