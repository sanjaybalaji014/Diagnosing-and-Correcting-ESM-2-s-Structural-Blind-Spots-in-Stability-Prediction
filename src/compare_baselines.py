"""
Phase 4: compare raw ESM-2, Person A's correction model, and the ThermoMPNN
structure-aware baseline against ground-truth ddG on the same held-out test
set mutations.

Handles two known mismatches between Person A's mutation numbering (derived
from the mega-scale construct sequence) and ThermoMPNN's numbering (derived
straight from the PDB file, which can carry expression-tag residues Person
A's numbering does not include):
  1. Per-protein position offset -- auto-detected the same way Phase 1's
     compute_position_offset() did, by aligning the two wild-type sequences.
  2. ThermoMPNN's ddG sign convention is flipped relative to this project's
     (confirmed by an initially strongly *negative* Spearman correlation) --
     corrected by negating thermompnn_ddg_pred before scoring.

Reads:
  data/processed/correction_model_results.csv
  data/processed/thermompnn_test_predictions.csv

Writes:
  data/processed/baseline_comparison_merged.csv
  data/processed/baseline_comparison_summary.csv
  data/processed/baseline_comparison_offsets.csv
"""
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CORRECTION_CSV = "data/processed/correction_model_results.csv"
THERMOMPNN_CSV = "data/processed/thermompnn_test_predictions.csv"
MERGED_OUT_CSV = "data/processed/baseline_comparison_merged.csv"
SUMMARY_OUT_CSV = "data/processed/baseline_comparison_summary.csv"
OFFSETS_OUT_CSV = "data/processed/baseline_comparison_offsets.csv"

MUT_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mut_type(mut_type):
    m = MUT_RE.match(str(mut_type).strip())
    if not m:
        return None, None, None
    wt_aa, position, mut_aa = m.group(1), int(m.group(2)), m.group(3)
    return wt_aa, position, mut_aa


def build_wt_sequence(df, pos_col, aa_col):
    """(min_pos, max_pos, position->aa dict, ordered sequence string)"""
    sub = df.drop_duplicates(subset=[pos_col]).sort_values(pos_col)
    positions = sub[pos_col].tolist()
    aas = sub[aa_col].tolist()
    pos_to_aa = dict(zip(positions, aas))
    lo, hi = min(positions), max(positions)
    seq = "".join(pos_to_aa.get(p, "X") for p in range(lo, hi + 1))
    return lo, seq


def find_offset(corr_lo, corr_seq, thermo_lo, thermo_seq):
    """
    Find the shift such that: thermo_position - shift == corr_position
    for the aligned region, using the longest common substring between the
    two wild-type sequences (mirrors Phase 1's offset-detection approach).
    """
    matcher = SequenceMatcher(None, thermo_seq, corr_seq, autojunk=False)
    block = matcher.find_longest_match(0, len(thermo_seq), 0, len(corr_seq))
    if block.size < 8:  # too little overlap to trust
        return None
    thermo_pos_at_block_start = thermo_lo + block.a
    corr_pos_at_block_start = corr_lo + block.b
    shift = thermo_pos_at_block_start - corr_pos_at_block_start
    return shift


def main():
    corr = pd.read_csv(CORRECTION_CSV)
    print(f"Loaded {len(corr)} rows from {CORRECTION_CSV}")

    parsed = corr["mut_type"].apply(parse_mut_type)
    corr["wt_aa"] = [p[0] for p in parsed]
    corr["position"] = [p[1] for p in parsed]
    corr["mut_aa"] = [p[2] for p in parsed]
    corr = corr.dropna(subset=["position"])
    corr["position"] = corr["position"].astype(int)
    corr = corr.rename(columns={"WT_name": "protein_id"})

    thermo = pd.read_csv(THERMOMPNN_CSV)
    print(f"Loaded {len(thermo)} rows from {THERMOMPNN_CSV}")
    # ThermoMPNN's ddG sign is flipped relative to this project's convention
    thermo["thermompnn_ddg_pred"] = -thermo["thermompnn_ddg_pred"]

    offset_rows = []
    merged_parts = []
    for pid in sorted(corr["protein_id"].unique()):
        c = corr[corr["protein_id"] == pid].copy()
        t = thermo[thermo["protein_id"] == pid].copy()
        if len(t) == 0:
            continue

        corr_lo, corr_seq = build_wt_sequence(c, "position", "wt_aa")
        thermo_lo, thermo_seq = build_wt_sequence(t, "position", "wt_aa")
        shift = find_offset(corr_lo, corr_seq, thermo_lo, thermo_seq)
        if shift is None:
            print(f"[WARN] {pid}: could not align sequences -- skipped")
            continue

        t["position_adj"] = t["position"] - shift
        m = c.merge(
            t[["position_adj", "wt_aa", "mut_aa", "thermompnn_ddg_pred"]],
            left_on=["position", "wt_aa", "mut_aa"],
            right_on=["position_adj", "wt_aa", "mut_aa"],
            how="inner",
        )
        offset_rows.append({
            "protein_id": pid, "shift": shift,
            "n_corr": len(c), "n_matched": len(m),
            "pct_matched": 100 * len(m) / len(c) if len(c) else 0,
        })
        merged_parts.append(m)

    offsets_df = pd.DataFrame(offset_rows)
    offsets_df.to_csv(OFFSETS_OUT_CSV, index=False)
    print("\nPer-protein offsets and match rates:")
    print(offsets_df.to_string(index=False))

    merged = pd.concat(merged_parts, ignore_index=True)
    print(f"\n{len(merged)} / {len(corr)} correction-model rows matched to a "
          f"ThermoMPNN prediction ({100 * len(merged) / len(corr):.1f}%)")
    merged.to_csv(MERGED_OUT_CSV, index=False)
    print(f"Wrote {MERGED_OUT_CSV}")

    methods = {
        "raw_esm2": "esm2_score_masked",
        "correction_model": "corrected_prediction",
        "thermompnn": "thermompnn_ddg_pred",
    }

    rows = []
    for name, col in methods.items():
        y_true = merged["ddG"].to_numpy()
        y_pred = merged[col].to_numpy()
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        yt, yp = y_true[valid], y_pred[valid]

        sp_r, _ = spearmanr(yt, yp)
        pe_r, _ = pearsonr(yt, yp)
        mae = np.mean(np.abs(yt - yp))

        rows.append({"method": name, "n": len(yt), "spearman_r": sp_r,
                      "pearson_r": pe_r, "mae": mae})
        print(f"\n{name} (n={len(yt)}):")
        print(f"  Spearman r = {sp_r:.4f}")
        print(f"  Pearson r  = {pe_r:.4f}")
        print(f"  MAE        = {mae:.4f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(SUMMARY_OUT_CSV, index=False)
    print(f"\nWrote {SUMMARY_OUT_CSV}")

    raw = summary.loc[summary["method"] == "raw_esm2"].iloc[0]
    corr_row = summary.loc[summary["method"] == "correction_model"].iloc[0]
    thermo_row = summary.loc[summary["method"] == "thermompnn"].iloc[0]

    for metric, better_is_higher in [("spearman_r", True), ("pearson_r", True), ("mae", False)]:
        gap = thermo_row[metric] - raw[metric]
        closed = corr_row[metric] - raw[metric]
        if not better_is_higher:
            gap, closed = -gap, -closed
        pct = 100 * closed / gap if gap != 0 else float("nan")
        print(f"\n{metric}: correction model closes {pct:.1f}% of the "
              f"raw-ESM2-to-ThermoMPNN gap")


if __name__ == "__main__":
    main()
