"""
Person B, Phase 2 Part 2: for each large-error mutation, check whether the
final-layer (33) probes could correctly predict the true structural feature
from ESM-2's embedding at that position.

  - Probe succeeds (correctly predicts structure) + ESM-2 still got stability
    wrong  -> "utilization failure": the info was there, ESM-2 didn't use it.
  - Probe fails (can't recover structure from the embedding either) -> ESM-2
    genuinely lacks that signal at this layer -> "content failure".

Reads:
  data/processed/esm2_errors.csv           (from Part 1)
  data/processed/multilayer_embeddings.npz (from Person A)
  trained_probes.pkl                       (from Person A — path below)

Writes:
  data/processed/utilization_analysis.csv  (one row per large-error mutation)
"""

import pickle

import numpy as np
import pandas as pd

ERRORS_CSV = "data/processed/esm2_errors.csv"
EMBEDDINGS_NPZ = "data/processed/multilayer_embeddings.npz"
PROBES_PKL = "trained_probes.pkl"  # adjust path if you put it elsewhere
OUTPUT_CSV = "data/processed/utilization_analysis.csv"

LAYER = 33  # final layer = the one actually used for stability prediction
BURIED_RSA_CUTOFF = 0.25  # must match structural_features.py's convention


def main():
    errors_df = pd.read_csv(ERRORS_CSV)
    large_error_df = errors_df[errors_df["is_large_error"]].copy()
    print(f"Loaded {len(errors_df)} total mutations; {len(large_error_df)} flagged large-error")

    with open(PROBES_PKL, "rb") as f:
        probes = pickle.load(f)
    layer_probes = probes[LAYER]
    print(f"Using layer {LAYER} probes: {list(layer_probes.keys())}")

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    npz_keys = data["keys"]  # (N, 2): [protein_id, position]
    layer_embeddings = data[f"layer_{LAYER}"]  # (N, 1280)

    # (protein_id, position) -> row index into layer_embeddings
    key_to_row = {(pid, int(pos)): i for i, (pid, pos) in enumerate(npz_keys)}
    print(f"Built embedding lookup for {len(key_to_row)} (protein_id, position) pairs")

    # true structural labels
    large_error_df["buried_true"] = (large_error_df["rsa"] < BURIED_RSA_CUTOFF).astype(int)

    # --- for the continuous target (contact_density), we need a "correct enough"
    # threshold. Compute the probe's typical error across ALL valid positions
    # first, then call a per-mutation prediction "correct" if its error is at
    # or below that typical (median) error. ---
    all_rows = np.array([key_to_row.get((pid, int(pos))) for pid, pos in
                          zip(errors_df["protein_id"], errors_df["position"])])
    valid_mask = all_rows != None  # noqa: E711
    valid_embeddings = layer_embeddings[all_rows[valid_mask].astype(int)]
    valid_true_cd = errors_df.loc[valid_mask, "contact_density"].to_numpy()
    valid_pred_cd = layer_probes["contact_density"].predict(valid_embeddings)
    valid_finite = np.isfinite(valid_true_cd) & np.isfinite(valid_pred_cd)
    median_abs_resid_cd = np.median(np.abs(valid_true_cd[valid_finite] - valid_pred_cd[valid_finite]))
    print(f"Median |residual| for contact_density probe (layer {LAYER}, all positions): "
          f"{median_abs_resid_cd:.3f}")

    results = []
    n_no_embedding = 0
    for _, row in large_error_df.iterrows():
        key = (row["protein_id"], int(row["position"]))
        row_idx = key_to_row.get(key)
        if row_idx is None:
            n_no_embedding += 1
            continue
        emb = layer_embeddings[row_idx].reshape(1, -1)

        pred_buried = int(layer_probes["buried"].predict(emb)[0])
        correct_buried = pred_buried == row["buried_true"]

        pred_ss = layer_probes["secondary_structure"].predict(emb)[0]
        correct_ss = (pred_ss == row["secondary_structure"]) if pd.notna(row["secondary_structure"]) else None

        pred_cd = float(layer_probes["contact_density"].predict(emb)[0])
        correct_cd = (abs(row["contact_density"] - pred_cd) <= median_abs_resid_cd
                       if pd.notna(row["contact_density"]) else None)

        checks = [c for c in (correct_buried, correct_ss, correct_cd) if c is not None]
        n_correct = sum(bool(c) for c in checks)
        verdict = "utilization_failure" if n_correct >= 2 else "content_failure"

        results.append({
            "protein_id": row["protein_id"],
            "position": row["position"],
            "wt_aa": row["wt_aa_x"],
            "mut_aa": row["mut_aa"],
            "error": row["error"],
            "abs_error": row["abs_error"],
            "buried_true": row["buried_true"],
            "buried_pred": pred_buried,
            "correct_buried": correct_buried,
            "secondary_structure_true": row["secondary_structure"],
            "secondary_structure_pred": pred_ss,
            "correct_secondary_structure": correct_ss,
            "contact_density_true": row["contact_density"],
            "contact_density_pred": pred_cd,
            "correct_contact_density": correct_cd,
            "n_correct_of_3": n_correct,
            "verdict": verdict,
        })

    if n_no_embedding:
        print(f"[WARN] {n_no_embedding} large-error mutations had no matching embedding "
              f"(protein_id/position not found in {EMBEDDINGS_NPZ}) — skipped")

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nAnalyzed {len(out_df)} large-error mutations")
    print(out_df["verdict"].value_counts())
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()