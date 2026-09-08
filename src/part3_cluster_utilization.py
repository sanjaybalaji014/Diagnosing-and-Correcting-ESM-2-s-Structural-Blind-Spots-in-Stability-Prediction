"""
Phase 3, Person B Part 2: for each failure cluster (burial x secondary_structure
x chem_transition) that Person A identified, compute what fraction of its
mutations are utilization failures vs. content failures.

Utilization/content verdict is computed per unique (protein_id, position) using
the layer-33 probes (same logic as part2_utilization_analysis.py), then joined
back onto every mutation at that position -- so all mutations sharing a
position share a verdict (burial/secondary_structure/contact_density are
properties of the position, not of the specific substitution; chem_transition
is what varies per-mutation and is what defines the clusters).

Reads:
  data/processed/df_with_error_and_strata.csv  (Person A's Phase 3 output)
  data/processed/multilayer_embeddings.npz
  trained_probes.pkl

Writes:
  data/processed/cluster_utilization_summary.csv   (all burial x ss x chem_transition combos, n>=30)
  data/processed/headline_finding_breakdown.csv     (buried charge-changing vs rest, by secondary structure)
"""

import pickle

import numpy as np
import pandas as pd

STRATA_CSV = "data/processed/df_with_error_and_strata.csv"
EMBEDDINGS_NPZ = "data/processed/multilayer_embeddings.npz"
PROBES_PKL = "trained_probes.pkl"
CLUSTER_OUT_CSV = "data/processed/cluster_utilization_summary.csv"
HEADLINE_OUT_CSV = "data/processed/headline_finding_breakdown.csv"

LAYER = 33
MIN_CLUSTER_N = 30
CHARGE_CHANGING = {"charged_to_nonpolar", "nonpolar_to_charged"}


def main():
    df = pd.read_csv(STRATA_CSV, low_memory=False)
    print(f"Loaded {len(df)} mutations from {STRATA_CSV}")

    with open(PROBES_PKL, "rb") as f:
        probes = pickle.load(f)
    layer_probes = probes[LAYER]

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    npz_keys = data["keys"]
    layer_embeddings = data[f"layer_{LAYER}"]
    key_to_row = {(pid, int(pos)): i for i, (pid, pos) in enumerate(npz_keys)}
    print(f"Embedding lookup built for {len(key_to_row)} (protein_id, position) pairs")

    # --- one verdict per unique (protein_id, position), reused across all
    # mutations at that position ---
    positions = df.drop_duplicates(subset=["protein_id", "position"])[
        ["protein_id", "position", "burial", "secondary_structure", "contact_density"]
    ].copy()

    row_idx = [key_to_row.get((pid, int(pos))) for pid, pos in
               zip(positions["protein_id"], positions["position"])]
    positions["row_idx"] = row_idx
    n_missing = positions["row_idx"].isna().sum()
    if n_missing:
        print(f"[WARN] {n_missing} positions have no matching embedding — dropped")
    positions = positions.dropna(subset=["row_idx"])
    positions["row_idx"] = positions["row_idx"].astype(int)

    emb = layer_embeddings[positions["row_idx"].to_numpy()]

    buried_true = (positions["burial"] == "buried").astype(int).to_numpy()
    pred_buried = layer_probes["buried"].predict(emb)
    correct_buried = pred_buried == buried_true

    ss_true = positions["secondary_structure"].to_numpy()
    pred_ss = layer_probes["secondary_structure"].predict(emb)
    correct_ss = pred_ss == ss_true

    cd_true = positions["contact_density"].to_numpy(dtype=float)
    pred_cd = layer_probes["contact_density"].predict(emb)
    cd_valid = np.isfinite(cd_true)
    median_abs_resid_cd = np.median(np.abs(cd_true[cd_valid] - pred_cd[cd_valid]))
    correct_cd = np.where(cd_valid, np.abs(cd_true - pred_cd) <= median_abs_resid_cd, np.nan)

    n_correct = (correct_buried.astype(float) + correct_ss.astype(float)
                 + np.nan_to_num(correct_cd, nan=0.0))
    n_checks = 2 + cd_valid.astype(float)  # buried+ss always checked, cd only if valid
    frac_correct = n_correct / n_checks
    positions["verdict"] = np.where(frac_correct >= (2 / 3), "utilization_failure", "content_failure")

    print(f"\nPosition-level verdicts ({len(positions)} unique positions):")
    print(positions["verdict"].value_counts())

    # join verdict back onto every mutation at that position
    df = df.merge(positions[["protein_id", "position", "verdict"]],
                   on=["protein_id", "position"], how="inner")
    print(f"\n{len(df)} mutations retained after joining verdicts")

    # --- full cluster table: burial x secondary_structure x chem_transition ---
    grouped = df.groupby(["burial", "secondary_structure", "chem_transition"])
    rows = []
    for (burial, ss, chem), g in grouped:
        if len(g) < MIN_CLUSTER_N:
            continue
        pct_util = (g["verdict"] == "utilization_failure").mean() * 100
        rows.append({
            "burial": burial,
            "secondary_structure": ss,
            "chem_transition": chem,
            "n": len(g),
            "mean_error": g["error"].mean(),
            "median_error": g["error"].median(),
            "pct_utilization_failure": pct_util,
            "pct_content_failure": 100 - pct_util,
        })
    cluster_df = pd.DataFrame(rows).sort_values("mean_error", ascending=False)
    cluster_df.to_csv(CLUSTER_OUT_CSV, index=False)
    print(f"\nWrote {len(cluster_df)} clusters (n>={MIN_CLUSTER_N}) to {CLUSTER_OUT_CSV}")
    print(cluster_df.head(15).to_string(index=False))

    # --- headline finding: buried + charge-changing vs everything else,
    # broken out by secondary structure ---
    df["is_headline_cluster"] = (df["burial"] == "buried") & df["chem_transition"].isin(CHARGE_CHANGING)

    headline_rows = []
    for ss, g_ss in df.groupby("secondary_structure"):
        for label, g in [("buried_charge_changing", g_ss[g_ss["is_headline_cluster"]]),
                          ("all_other_buried", g_ss[(g_ss["burial"] == "buried") & ~g_ss["is_headline_cluster"]])]:
            if len(g) == 0:
                continue
            pct_util = (g["verdict"] == "utilization_failure").mean() * 100
            headline_rows.append({
                "secondary_structure": ss,
                "group": label,
                "n": len(g),
                "mean_error": g["error"].mean(),
                "pct_utilization_failure": pct_util,
                "pct_content_failure": 100 - pct_util,
            })
    headline_df = pd.DataFrame(headline_rows)
    headline_df.to_csv(HEADLINE_OUT_CSV, index=False)
    print(f"\n=== Headline finding breakdown (wrote {HEADLINE_OUT_CSV}) ===")
    print(headline_df.to_string(index=False))

    overall = df[df["is_headline_cluster"]]
    print(f"\nOverall: buried charge-changing mutations (n={len(overall)}): "
          f"{(overall['verdict'] == 'utilization_failure').mean() * 100:.1f}% utilization failure")


if __name__ == "__main__":
    main()