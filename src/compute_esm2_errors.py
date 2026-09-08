"""
Person B, Phase 2 Part 1: define and compute ESM-2's prediction error per
mutation, and flag "large error" mutations.

Reads:  data/processed/esm2_structural_final.csv
Writes: data/processed/esm2_errors.csv  (adds predicted_ddg, error, abs_error,
        is_large_error columns to every row)

Method:
  ESM-2's masked-marginal score and real ddG are on different scales/units
  (log-odds vs. energy), so a raw difference between them isn't a meaningful
  "error." Instead we fit a simple linear calibration ddG ~ esm2_score_masked
  across the whole dataset (least squares), and define each mutation's error
  as its residual from that fit: actual ddG - predicted ddG. Large-error
  mutations are the top 20% by |error| (configurable below).
"""

import numpy as np
import pandas as pd

INPUT_CSV = "data/processed/esm2_structural_final.csv"
OUTPUT_CSV = "data/processed/esm2_errors.csv"
LARGE_ERROR_PERCENTILE = 80  # top 20% by |error|

TARGET_COL = "ddG"
ESM2_COL = "esm2_score_masked"


def main():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")

    before = len(df)
    df = df.dropna(subset=[TARGET_COL, ESM2_COL])
    print(f"Dropped {before - len(df)} rows missing {TARGET_COL!r} or {ESM2_COL!r}; {len(df)} remain")

    # --- sanity check: correlation direction, before assuming anything ---
    pearson_r = df[ESM2_COL].corr(df[TARGET_COL], method="pearson")
    spearman_r = df[ESM2_COL].corr(df[TARGET_COL], method="spearman")
    print(f"\nCorrelation between {ESM2_COL} and {TARGET_COL}:")
    print(f"  Pearson r:  {pearson_r:.4f}")
    print(f"  Spearman r: {spearman_r:.4f}")
    if pearson_r < 0:
        print("  -> negative correlation: higher ESM-2 score associates with LOWER ddG "
              "here. The linear fit below handles this automatically (sign is absorbed "
              "into the fitted slope), but worth knowing when you interpret results.")

    # --- calibrate: fit ddG ~ esm2_score_masked (simple OLS, 1 feature) ---
    x = df[ESM2_COL].to_numpy()
    y = df[TARGET_COL].to_numpy()
    slope, intercept = np.polyfit(x, y, deg=1)
    print(f"\nLinear calibration: ddG_pred = {slope:.4f} * {ESM2_COL} + {intercept:.4f}")

    df["predicted_ddg"] = slope * x + intercept
    df["error"] = df[TARGET_COL] - df["predicted_ddg"]
    df["abs_error"] = df["error"].abs()

    threshold = np.percentile(df["abs_error"], LARGE_ERROR_PERCENTILE)
    df["is_large_error"] = df["abs_error"] >= threshold
    print(f"\nLarge-error threshold (top {100 - LARGE_ERROR_PERCENTILE}% by |error|): "
          f"|error| >= {threshold:.4f}")
    print(f"Flagged {df['is_large_error'].sum()} / {len(df)} mutations as large-error "
          f"({df['is_large_error'].mean() * 100:.1f}%)")

    KEEP_COLS = [
        "protein_id", "position", "wt_aa_x", "mut_aa", "ddG", "esm2_score_masked",
        "rsa", "secondary_structure", "contact_density", "dist_to_core",
        "predicted_ddg", "error", "abs_error", "is_large_error",
    ]
    df[KEEP_COLS].to_csv(OUTPUT_CSV, index=False)    
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()