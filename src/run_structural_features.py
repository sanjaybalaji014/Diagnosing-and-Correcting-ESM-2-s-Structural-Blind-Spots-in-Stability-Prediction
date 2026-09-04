"""
Run the Person B structural-features pipeline end-to-end on person_b_input.csv.

Usage (from the repo root, with your venv active):
    python src/run_structural_features.py

Reads:  data/processed/person_b_input.csv
Writes: data/processed/structural_features.csv
        structures/  (downloaded PDB files, cached so re-runs are fast)
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structural_features import build_structural_feature_table

INPUT_CSV = os.path.join("data", "processed", "person_b_input.csv")
OUTPUT_CSV = os.path.join("data", "processed", "structural_features.csv")
STRUCTURE_DIR = "structures"


def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(
            f"Can't find {INPUT_CSV} — run this script from the repo root "
            f"(the folder containing data/, src/, structures/)."
        )

    mutations_df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(mutations_df)} mutations across "
          f"{mutations_df['protein_id'].nunique()} proteins from {INPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    feat_df = build_structural_feature_table(mutations_df, structure_dir=STRUCTURE_DIR)
    feat_df.to_csv(OUTPUT_CSV, index=False)

    n = len(feat_df)
    coverage = feat_df["rsa"].notna().mean() * 100
    print(f"\nWrote {n} rows to {OUTPUT_CSV}")
    print(f"Structural feature coverage (non-NaN rsa): {coverage:.1f}%")

    per_protein = feat_df.groupby("protein_id").apply(
        lambda g: pd.Series({
            "n_mutations": len(g),
            "n_valid_rsa": g["rsa"].notna().sum(),
            "n_nan_rsa": g["rsa"].isna().sum(),
        })
    ).reset_index()
    per_protein["status"] = per_protein.apply(
        lambda r: "OK" if r["n_nan_rsa"] == 0 else ("ZERO" if r["n_valid_rsa"] == 0 else "PARTIAL"),
        axis=1,
    )
    per_protein = per_protein.sort_values("n_nan_rsa", ascending=False)
    per_protein.to_csv(os.path.join("data", "processed", "structural_features_report.csv"), index=False)
    print("\nPer-protein report saved to data/processed/structural_features_report.csv")
    print(per_protein[per_protein["status"] != "OK"].to_string(index=False))


if __name__ == "__main__":
    main()