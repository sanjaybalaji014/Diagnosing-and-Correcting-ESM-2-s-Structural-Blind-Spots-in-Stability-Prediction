from __future__ import annotations

import numpy as np
import pandas as pd

MERGE_KEYS = ["protein_id", "position", "wt_aa", "mut_aa"]


def merge_esm2_and_structural(esm2_csv: str, structural_csv: str) -> pd.DataFrame:
    esm2_df = pd.read_csv(esm2_csv)
    struct_df = pd.read_csv(structural_csv)

    missing_esm2 = set(MERGE_KEYS) - set(esm2_df.columns)
    missing_struct = set(MERGE_KEYS) - set(struct_df.columns)
    if missing_esm2:
        raise ValueError(f"esm2_csv missing merge key columns: {missing_esm2}")
    if missing_struct:
        raise ValueError(f"structural_csv missing merge key columns: {missing_struct}")

    merged = esm2_df.merge(struct_df, on=MERGE_KEYS, how="inner", suffixes=("", "_struct"))

    n_esm2, n_struct, n_merged = len(esm2_df), len(struct_df), len(merged)
    print(f"ESM-2 rows: {n_esm2} | structural rows: {n_struct} | merged (inner join): {n_merged}")
    if n_merged < min(n_esm2, n_struct):
        dropped = min(n_esm2, n_struct) - n_merged
        print(f"[WARN] {dropped} rows dropped in the merge - check for mismatched "
              f"protein_id/position/wt_aa/mut_aa naming or indexing (0- vs 1-based) "
              f"between the two tracks.")
    return merged


def make_protein_split(df: pd.DataFrame, seed: int = 42, test_frac: float = 0.2) -> pd.DataFrame:
    proteins = df["protein_id"].unique()
    rng = np.random.RandomState(seed)
    proteins = proteins.copy()
    rng.shuffle(proteins)
    n_test = int(test_frac * len(proteins))
    test_proteins = set(proteins[:n_test])
    df = df.copy()
    df["split"] = df["protein_id"].apply(lambda p: "test" if p in test_proteins else "train")
    print(f"{len(proteins)} proteins -> {len(test_proteins)} test / {len(proteins) - len(test_proteins)} train "
          f"({df['split'].value_counts().to_dict()} rows)")
    return df


def save_split_assignment(df: pd.DataFrame, out_csv: str = "data/processed/protein_split.csv"):
    split_df = df[["protein_id", "split"]].drop_duplicates().sort_values("protein_id")
    split_df.to_csv(out_csv, index=False)
    print(f"Saved protein-level split assignment to {out_csv} ({len(split_df)} proteins)")
    return split_df
'@ | Out-File -FilePath src\merge_and_split.py -Encoding utf8