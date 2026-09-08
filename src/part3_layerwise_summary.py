"""
Person B, Phase 2 Part 3: layer-wise summary — at which layer does ESM-2 best
encode each structural feature, and does that peak line up with the final
layer (33, used for stability prediction) or does structural awareness get
discarded somewhere along the way?

Evaluates every trained probe against ALL 5537 positions (not just
large-error mutations) at every available layer.

Reads:
  data/processed/multilayer_embeddings.npz
  trained_probes.pkl

Writes:
  data/processed/layerwise_probe_performance.csv
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

EMBEDDINGS_NPZ = "data/processed/multilayer_embeddings.npz"
PROBES_PKL = "trained_probes.pkl"
STRUCTURAL_FEATURES_CSV = "data/processed/structural_features.csv"
OUTPUT_CSV = "data/processed/layerwise_probe_performance.csv"

BURIED_RSA_CUTOFF = 0.25


def main():
    with open(PROBES_PKL, "rb") as f:
        probes = pickle.load(f)
    layers = sorted(probes.keys())
    print(f"Layers available: {layers}")

    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    npz_keys = data["keys"]  # (N, 2): [protein_id, position]
    print(f"{len(npz_keys)} (protein_id, position) pairs in embeddings file")

    # true structural values, deduplicated to one row per (protein_id, position)
    # since structural_features.csv is per-mutation (multiple mut_aa per position)
    struct_df = pd.read_csv(STRUCTURAL_FEATURES_CSV)
    struct_df = struct_df.drop_duplicates(subset=["protein_id", "position"])
    struct_df["buried_true"] = (struct_df["rsa"] < BURIED_RSA_CUTOFF).astype(int)
    struct_lookup = struct_df.set_index(["protein_id", "position"])[
        ["buried_true", "secondary_structure", "contact_density"]
    ].to_dict("index")

    true_buried, true_ss, true_cd = [], [], []
    keep_idx = []
    for i, (pid, pos) in enumerate(npz_keys):
        info = struct_lookup.get((pid, int(pos)))
        if info is None:
            continue
        keep_idx.append(i)
        true_buried.append(info["buried_true"])
        true_ss.append(info["secondary_structure"])
        true_cd.append(info["contact_density"])
    keep_idx = np.array(keep_idx)
    true_buried = np.array(true_buried)
    true_ss = np.array(true_ss, dtype=object)
    true_cd = np.array(true_cd, dtype=float)
    print(f"Matched {len(keep_idx)} / {len(npz_keys)} positions to structural_features.csv")

    rows = []
    for layer in layers:
        emb = data[f"layer_{layer}"][keep_idx]
        layer_probes = probes[layer]

        pred_buried = layer_probes["buried"].predict(emb)
        acc_buried = accuracy_score(true_buried, pred_buried)

        ss_mask = pd.notna(true_ss)
        pred_ss = layer_probes["secondary_structure"].predict(emb[ss_mask])
        acc_ss = accuracy_score(true_ss[ss_mask], pred_ss)

        cd_mask = np.isfinite(true_cd)
        pred_cd = layer_probes["contact_density"].predict(emb[cd_mask])
        r2_cd = r2_score(true_cd[cd_mask], pred_cd)

        rows.append({
            "layer": layer,
            "buried_accuracy": acc_buried,
            "secondary_structure_accuracy": acc_ss,
            "contact_density_r2": r2_cd,
        })
        print(f"layer {layer:>3}: buried_acc={acc_buried:.3f}  "
              f"ss_acc={acc_ss:.3f}  contact_density_r2={r2_cd:.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_CSV, index=False)

    print("\nPeak layer per feature:")
    for col in ["buried_accuracy", "secondary_structure_accuracy", "contact_density_r2"]:
        best = result_df.loc[result_df[col].idxmax()]
        flag = "" if int(best["layer"]) == 33 else "  <-- NOT the final layer (33)"
        print(f"  {col}: layer {int(best['layer'])} ({best[col]:.3f}){flag}")

    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()