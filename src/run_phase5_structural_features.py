"""
Phase 5: structural features for the ClinVar/gnomAD clinical-variant set.

Unlike Phase 1's ~100 Mega-scale proteins (small, single-domain, each with a
clean experimental PDB), Phase 5's 10 genes (APC, BRCA1, BRCA2, MLH1, MSH2,
MSH6, PTEN, RB1, TP53, VHL) are large, multi-domain, and mostly have no
single PDB structure covering the full sequence -- so this pulls AlphaFold DB
predicted structures by UniProt accession instead.

Two wrinkles Phase 1 didn't have to handle, both dealt with here:

  1. AlphaFold DB fragments long proteins (>2700 residues) into multiple
     overlapping files (-F1-, -F2-, ...). APC (2843 aa) and BRCA2 (3418 aa)
     both exceed that threshold, so a naive single-fragment fetch (what
     structural_features.fetch_alphafold_structure does) silently misses
     every variant past ~residue 1400-2700. This script fetches *all*
     fragments for each UniProt ID and merges their residue tables.

  2. AlphaFold DB numbering follows the canonical UniProt isoform 1:1 (no
     expression tags / crystallization artifacts the way PDB files have), so
     the position offset that Phase 1 and Phase 4 both had to auto-detect
     should be 0 here. This script still checks: for every variant it
     compares the CSV's wt_aa against the residue actually found at that
     position in the structure, and reports the per-gene mismatch rate, so a
     wrong UniProt accession or isoform gets caught instead of silently
     producing garbage features.

Reads:
  data/processed/phase5_esm2_scored.csv (gene, wt_aa, position, mut_aa,
    label, protein_id, aligned, wt_seq, truncated, esm2_score -- Person A's
    output). 758 rows: Person A already dropped 25 benign rows whose wt_aa
    didn't align to their own reference sequence -- 22 of those are exactly
    the wt_aa-vs-AlphaFold-structure mismatches this script independently
    flagged in the prior 783-row input (see phase5_wt_aa_mismatches.csv from
    that run), which cross-validates both checks. This script still re-runs
    its own structure-based wt_aa check below rather than trusting
    `aligned` blindly, since it's a check against a different reference
    (AlphaFold's canonical sequence vs whatever Person A aligned against).

Writes:
  data/processed/phase5_structural_features.csv  (adds rsa,
    secondary_structure, contact_density, dist_to_core, plddt,
    plddt_confidence to every input row; esm2_score/wt_seq/aligned/truncated
    carried through unchanged for the downstream correction-model step)
  data/processed/phase5_structural_features_report.csv
  data/processed/phase5_wt_aa_mismatches.csv
  structures/  (downloaded AlphaFold fragments, cached so re-runs are fast)

Usage (from the repo root, with your venv active):
    python src/run_phase5_structural_features.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structural_features import (
    THREE_TO_ONE,
    build_residue_table,
    compute_contact_density,
    compute_dist_to_core,
    get_rsa_and_ss,
)
import biotite.structure as struc
import biotite.structure.io.pdb as pdb_io

INPUT_CSV = os.path.join("data", "processed", "phase5_esm2_scored.csv")
OUTPUT_CSV = os.path.join("data", "processed", "phase5_structural_features.csv")
REPORT_CSV = os.path.join("data", "processed", "phase5_structural_features_report.csv")
MISMATCH_CSV = os.path.join("data", "processed", "phase5_wt_aa_mismatches.csv")
STRUCTURE_DIR = "structures"

# Canonical human UniProt accessions for the 10 Phase 5 genes.
GENE_TO_UNIPROT = {
    "VHL": "P40337",
    "PTEN": "P60484",
    "TP53": "P04637",
    "BRCA1": "P38398",
    "BRCA2": "P51587",
    "MLH1": "P40692",
    "MSH2": "P43246",
    "MSH6": "P52701",
    "APC": "P25054",
    "RB1": "P06400",
}


def fetch_alphafold_fragments(uniprot_id: str, out_dir: str) -> list[str]:
    """
    Download every AlphaFold DB fragment for this UniProt accession.

    Queries AlphaFold's own prediction API (rather than guessing a
    -F1-/-F2-/... filename and a model version number directly) so this
    keeps working across model-version bumps (confirmed live: this API
    reported model_v6 while the old hardcoded fetch was still requesting
    model_v4 and silently 404ing on every single gene).

    Filters the returned entries down to genuine positional fragments only.
    Confirmed live (TP53) that the API also returns alternate splice-isoform
    entries under the same accession (delta133p53, p53beta, p53gamma, ...) --
    these all report sequenceStart=1 like the canonical entry, but are named
    "AF-{uniprot}-{n}-F1" (an extra digit BEFORE "F1"), whereas a genuine
    positional fragment of a long protein is named "AF-{uniprot}-F2",
    "AF-{uniprot}-F3" (digit AFTER "F", no extra prefix). Treating isoform
    entries as if they were fragment continuations was letting mismatched
    residues leak into the merged table wherever the canonical entry didn't
    cover a given position first -- this is what caused the wt_aa mismatches
    seen for TP53/PTEN/MSH2/RB1/BRCA1 in the first working run.
    """
    os.makedirs(out_dir, exist_ok=True)

    import glob
    cached = sorted(glob.glob(os.path.join(out_dir, f"AF-{uniprot_id}-F*.pdb")))
    # Only trust the cache if it's just the canonical entry, or numbered
    # continuations (F1, F2, F3...) -- not the doubled "AF-AF-..." files an
    # earlier, buggy version of this script could have left behind.
    cached = [p for p in cached if re.match(rf"^AF-{re.escape(uniprot_id)}-F\d+\.pdb$", os.path.basename(p))]

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        r = requests.get(api_url, timeout=30)
    except requests.RequestException as e:
        if cached:
            print(f"  [WARN] AlphaFold API unreachable for {uniprot_id} ({e}) -- "
                  f"using {len(cached)} already-cached fragment file(s) instead")
            return cached
        print(f"  [WARN] AlphaFold API request failed for {uniprot_id}: {e}")
        return []
    if r.status_code != 200:
        print(f"  [WARN] AlphaFold API returned {r.status_code} for {uniprot_id} "
              f"-- this UniProt accession is not indexed in AlphaFold DB "
              f"(confirmed: BRCA2/APC exceed AlphaFold DB's proteome-wide "
              f"length coverage; verified via both this API and a direct "
              f"file fetch, both 404).")
        return []

    entries = r.json()
    if not entries:
        return []

    fragment_re = re.compile(rf"^AF-{re.escape(uniprot_id)}-F\d+$")
    fragments = [e for e in entries if fragment_re.match(e.get("modelEntityId", ""))]
    skipped = len(entries) - len(fragments)
    if skipped:
        print(f"  ({skipped} alternate isoform/variant entr{'y' if skipped == 1 else 'ies'} "
              f"under this accession excluded -- not positional fragments)")

    paths = []
    for entry in fragments:
        entity_id = entry["modelEntityId"]
        pdb_url = entry.get("pdbUrl")
        if not pdb_url:
            continue
        out_path = os.path.join(out_dir, f"{entity_id}.pdb")
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        pr = requests.get(pdb_url, timeout=30)
        if pr.status_code == 200:
            with open(out_path, "w") as f:
                f.write(pr.text)
            paths.append(out_path)
        else:
            print(f"  [WARN] fetch failed ({pr.status_code}) for {pdb_url}")
    return paths


def load_atom_array_with_confidence(structure_path: str, chain_id: str = "A"):
    """Like structural_features.load_atom_array, but also pulls per-atom
    b_factor -- for an AlphaFold PDB file this field holds pLDDT (0-100),
    AlphaFold's own per-residue confidence score, not a real crystallographic
    B-factor (confirmed live: VHL's known-disordered N-terminal tail reads
    43-62 here, consistent with AlphaFold's own reported low-confidence
    region for that segment). biotite doesn't load b_factor by default, so
    this passes extra_fields explicitly rather than reusing the shared
    Phase 1 loader (which real PDB structures don't need this for)."""
    pdb_file = pdb_io.PDBFile.read(structure_path)
    atom_array = pdb_file.get_structure(model=1, extra_fields=["b_factor"])
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    if chain_id:
        mask = atom_array.chain_id == chain_id
        if mask.any():
            atom_array = atom_array[mask]
    return atom_array


def plddt_confidence_band(plddt: float) -> str:
    """AlphaFold's own official pLDDT bins."""
    if np.isnan(plddt):
        return None
    if plddt < 50:
        return "very_low"
    if plddt < 70:
        return "low"
    if plddt < 90:
        return "confident"
    return "very_high"


def build_merged_residue_table(fragment_paths: list[str]) -> tuple[list[dict], dict[int, float]]:
    """Load every fragment, build its residue table, and merge into one
    table keyed by UniProt residue number (fragments overlap; first
    occurrence wins, which is fine since overlap regions are near-identical
    predictions of the same sequence). Also returns a {res_id: plddt} lookup
    built from the CA atom's b_factor in each fragment, same merge rule."""
    merged: dict[int, dict] = {}
    plddt_lookup: dict[int, float] = {}
    for path in fragment_paths:
        atom_array = load_atom_array_with_confidence(path, chain_id="A")
        for row in build_residue_table(atom_array):
            merged.setdefault(row["res_id"], row)
        ca_mask = atom_array.atom_name == "CA"
        for res_id, b_factor in zip(atom_array.res_id[ca_mask], atom_array.b_factor[ca_mask]):
            plddt_lookup.setdefault(int(res_id), float(b_factor))
    return [merged[k] for k in sorted(merged)], plddt_lookup


def main():
    if not os.path.exists(INPUT_CSV):
        raise SystemExit(
            f"Can't find {INPUT_CSV} -- run `git pull` first, then run this "
            f"script from the repo root."
        )

    variants = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(variants)} variants across {variants['gene'].nunique()} "
          f"genes from {INPUT_CSV}")

    missing_genes = sorted(set(variants["gene"]) - set(GENE_TO_UNIPROT))
    if missing_genes:
        raise SystemExit(
            f"No UniProt accession mapped for gene(s): {missing_genes} -- "
            f"add them to GENE_TO_UNIPROT before running."
        )

    residue_tables: dict[str, list[dict]] = {}
    plddt_tables: dict[str, dict[int, float]] = {}
    for gene, uniprot_id in GENE_TO_UNIPROT.items():
        if gene not in set(variants["gene"]):
            continue
        print(f"\n{gene} ({uniprot_id}): fetching AlphaFold fragment(s)...")
        fragment_paths = fetch_alphafold_fragments(uniprot_id, STRUCTURE_DIR)
        if not fragment_paths:
            print(f"[WARN] {gene}: no AlphaFold structure found for {uniprot_id}")
            residue_tables[gene] = []
            plddt_tables[gene] = {}
            continue
        print(f"  {len(fragment_paths)} fragment(s): "
              f"{[os.path.basename(p) for p in fragment_paths]}")
        residue_tables[gene], plddt_tables[gene] = build_merged_residue_table(fragment_paths)
        max_res = max((r["res_id"] for r in residue_tables[gene]), default=0)
        print(f"  merged residue table covers positions 1-{max_res}")

    rows = []
    mismatch_rows = []
    for _, v in variants.iterrows():
        gene = v["gene"]
        position = int(v["position"])
        wt_aa, mut_aa, label = v["wt_aa"], v["mut_aa"], v["label"]
        table = residue_tables.get(gene, [])
        plddt_lookup = plddt_tables.get(gene, {})

        rsa, ss = get_rsa_and_ss(table, position)
        contact_density = compute_contact_density(table, position)
        dist_to_core = compute_dist_to_core(table, position)
        plddt = plddt_lookup.get(position, np.nan)
        confidence = plddt_confidence_band(plddt)

        # QC: does the structure's residue at this position actually match
        # the CSV's wt_aa? (Catches wrong isoform / wrong UniProt ID -- and,
        # confirmed live, a real numbering mismatch specific to the gnomAD
        # benign-proxy subset: every one of the mismatches found in this
        # dataset was a `benign` row, never a `pathogenic` one, across every
        # affected gene. That's not noise -- it means the gnomAD extraction
        # is very likely keyed to a different reference transcript/isoform
        # than the ClinVar extraction for these genes. A row that fails
        # this check is scored against the WRONG residue, so its structural
        # features are nulled out below rather than kept.)
        struct_row = next((r for r in table if r["res_id"] == position), None)
        struct_aa = THREE_TO_ONE.get(struct_row["res_name"]) if struct_row else None
        wt_match = (struct_aa == wt_aa) if struct_aa is not None else None
        mismatch_rows.append({
            "gene": gene, "position": position, "csv_wt_aa": wt_aa,
            "structure_aa": struct_aa, "match": wt_match, "label": label,
        })

        if wt_match is False:
            rsa, ss, contact_density, dist_to_core = np.nan, None, np.nan, np.nan
            plddt, confidence = np.nan, None

        rows.append({
            "protein_id": v["protein_id"], "gene": gene, "position": position,
            "wt_aa": wt_aa, "mut_aa": mut_aa, "label": label,
            "rsa": rsa, "secondary_structure": ss,
            "contact_density": contact_density, "dist_to_core": dist_to_core,
            "plddt": plddt, "plddt_confidence": confidence,
            "wt_aa_matches_structure": wt_match,
            # carried through from Person A's ESM-2 scoring step
            "esm2_score": v.get("esm2_score", np.nan),
            "wt_seq": v.get("wt_seq"),
            "aligned": v.get("aligned"),
            "truncated": v.get("truncated"),
        })

    feat_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    feat_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(feat_df)} rows to {OUTPUT_CSV}")

    coverage = feat_df["rsa"].notna().mean() * 100
    print(f"Structural feature coverage (non-NaN rsa): {coverage:.1f}%")

    conf_counts = feat_df["plddt_confidence"].value_counts(dropna=False)
    print(f"\npLDDT confidence bands (AlphaFold's own bins): {conf_counts.to_dict()}")
    low_conf_n = int(feat_df["plddt_confidence"].isin(["low", "very_low"]).sum())
    if low_conf_n:
        print(f"[NOTE] {low_conf_n} variants fall in AlphaFold's low/very_low pLDDT bands "
              f"(<70) -- features are still reported (not nulled), but treat RSA/burial at "
              f"these positions cautiously: low pLDDT usually means a disordered or "
              f"poorly-predicted region, where 'buried vs exposed' is less meaningful. "
              f"See the plddt_confidence column to filter or weight these downstream.")

    if "truncated" in feat_df.columns:
        n_truncated = int(feat_df["truncated"].fillna(False).sum())
        if n_truncated:
            print(f"[NOTE] {n_truncated} variants used a truncated ESM-2 input sequence "
                  f"(Person A's `truncated` flag, carried through unchanged) -- their "
                  f"esm2_score may reflect incomplete sequence context, independent of "
                  f"the structural features computed here.")

    mismatch_df = pd.DataFrame(mismatch_rows)
    only_mismatches = mismatch_df[mismatch_df["match"] == False].sort_values(["gene", "position"])
    only_mismatches.to_csv(MISMATCH_CSV, index=False)
    print(f"\nWrote {len(only_mismatches)} wt_aa mismatches to {MISMATCH_CSV} for inspection "
          f"(their structural features are set to NaN, not silently kept)")
    if len(only_mismatches):
        by_label = only_mismatches["label"].value_counts()
        print(f"Mismatches by label: {by_label.to_dict()}")
        if by_label.get("pathogenic", 0) == 0 and by_label.get("benign", 0) > 0:
            print(
                "[FLAG] Every mismatch is a `benign` (gnomAD) row -- this points to a "
                "real transcript/isoform numbering difference between the ClinVar "
                "pathogenic pull and the gnomAD benign-proxy pull, not a structural-"
                "pipeline bug. Worth raising with whoever built the gnomAD extraction "
                "before trusting the benign class's positions at face value."
            )

    per_gene = feat_df.groupby("gene").apply(
        lambda g: pd.Series({
            "n_variants": len(g),
            "n_valid_rsa": g["rsa"].notna().sum(),
            "n_wt_match": (g["wt_aa_matches_structure"] == True).sum(),
            "n_wt_mismatch": (g["wt_aa_matches_structure"] == False).sum(),
            "n_wt_no_structure": g["wt_aa_matches_structure"].isna().sum(),
        }),
        include_groups=False,
    ).reset_index()
    per_gene["wt_match_rate_pct"] = (
        100 * per_gene["n_wt_match"] / per_gene["n_variants"]
    ).round(1)
    per_gene.to_csv(REPORT_CSV, index=False)
    print(f"\nPer-gene report saved to {REPORT_CSV}")
    print(per_gene.to_string(index=False))

    low_match = per_gene[per_gene["wt_match_rate_pct"] < 90]
    if len(low_match):
        print(
            f"\n[WARN] These genes have <90% wt_aa-vs-structure match -- likely "
            f"wrong UniProt accession, wrong isoform, or a fragment-numbering "
            f"issue. Double-check GENE_TO_UNIPROT for these before trusting "
            f"their features:\n{low_match[['gene', 'wt_match_rate_pct']].to_string(index=False)}"
        )

    no_structure_genes = sorted(g for g, t in residue_tables.items() if not t)
    if no_structure_genes:
        n_excluded = int(variants["gene"].isin(no_structure_genes).sum())
        print(
            f"\n[NOTE] No structure available for: {no_structure_genes} "
            f"({n_excluded} variants, {100 * n_excluded / len(variants):.1f}% of the "
            f"dataset) -- confirmed these UniProt accessions are not indexed in "
            f"AlphaFold DB (both the prediction API and a direct file fetch "
            f"returned 404). These proteins exceed AlphaFold DB's proteome-wide "
            f"length coverage. Document this as an explicit exclusion in the "
            f"paper rather than silently dropping rows -- the ROC-AUC analysis "
            f"should report it ran on the remaining genes/variants only.")


if __name__ == "__main__":
    main()
