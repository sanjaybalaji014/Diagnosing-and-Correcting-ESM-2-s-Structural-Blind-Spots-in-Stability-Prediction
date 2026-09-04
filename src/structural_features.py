"""
Person B track — Structural Features pipeline
================================================

Given a list of mutations (protein_id, position, wt_aa, mut_aa), this module:
  1. Fetches a structure for each protein_id (PDB if available, else AlphaFold DB,
     else falls back to ESMFold prediction).
  2. Computes relative solvent accessibility (RSA) and secondary structure —
     using biotite's built-in SASA (Shrake-Rupley) and SSE (P-SEA) algorithms,
     both pure-Python/compiled-wheel, so there's no external DSSP binary to
     install. This runs natively on Windows/Mac/Linux with just `pip install`.
  3. Computes contact density (# CA atoms within a cutoff of the mutated
     residue's CA).
  4. Computes distance to the hydrophobic core (centroid of buried hydrophobic
     residues, or geometric center as a fallback).
  5. Attaches a conservation score if a precomputed source is available (else
     NaN, flagged as a limitation).

Output: one row per mutation with columns
  protein_id, position, wt_aa, mut_aa, rsa, secondary_structure,
  contact_density, dist_to_core, conservation_score

Install deps (works natively on Windows — no DSSP/WSL needed):

    pip install biotite pandas requests numpy

Usage (once you have your mutation list as a DataFrame):

    from structural_features import build_structural_feature_table
    feat_df = build_structural_feature_table(mutations_df, structure_dir="structures/")
    feat_df.to_csv("data/processed/structural_features.csv", index=False)
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

import biotite.structure as struc
import biotite.structure.io.pdb as pdb

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BURIED_RSA_CUTOFF = 0.25  # 20-25% RSA is the common buried/exposed threshold
CONTACT_CUTOFF_ANGSTROM = 8.0  # CA-CA distance cutoff for contact density
HYDROPHOBIC_AAS_3LETTER = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "CYS"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# Theoretical max solvent-accessible surface area per residue, in Å²
# (Tien et al. 2013, "theoretical" column) — used to normalize raw SASA -> RSA.
MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}


# ---------------------------------------------------------------------------
# Step 1: Fetch structures
# ---------------------------------------------------------------------------

def fetch_pdb_structure(pdb_id: str, out_dir: str) -> Optional[str]:
    """Download a PDB structure file by 4-char PDB ID. Returns local file path or None."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{pdb_id}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(out_path, "w") as f:
            f.write(r.text)
        return out_path
    return None


def fetch_alphafold_structure(uniprot_id: str, out_dir: str) -> Optional[str]:
    """Download an AlphaFold DB predicted structure (v4) by UniProt accession."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"AF-{uniprot_id}.pdb")
    if os.path.exists(out_path):
        return out_path
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        with open(out_path, "w") as f:
            f.write(r.text)
        return out_path
    return None


def fold_with_esmfold(sequence: str, protein_id: str, out_dir: str) -> Optional[str]:
    """
    Fallback: predict structure with ESMFold via the public API (good for short
    sequences; rate-limited). Only use this for proteins with no PDB/AlphaFold
    hit — it's the slowest option, and coordinate with Person A on the sampled
    protein set first so you're not folding things unnecessarily.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{protein_id}_esmfold.pdb")
    if os.path.exists(out_path):
        return out_path
    try:
        r = requests.post(
            "https://api.esmatlas.com/foldSequence/v1/pdb/",
            data=sequence,
            timeout=120,
        )
        if r.status_code == 200 and r.text.startswith(("HEADER", "ATOM")):
            with open(out_path, "w") as f:
                f.write(r.text)
            return out_path
    except requests.RequestException:
        pass
    return None


def get_structure_path(
    protein_id: str,
    out_dir: str,
    pdb_id: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    sequence: Optional[str] = None,
) -> Optional[str]:
    """Try PDB -> AlphaFold -> ESMFold, in that order, and return whichever succeeds."""
    if pdb_id:
        path = fetch_pdb_structure(pdb_id, out_dir)
        if path:
            return path
    if uniprot_id:
        path = fetch_alphafold_structure(uniprot_id, out_dir)
        if path:
            return path
    if sequence:
        path = fold_with_esmfold(sequence, protein_id, out_dir)
        if path:
            return path
    return None


# ---------------------------------------------------------------------------
# Step 2: load structure + build a per-residue feature table (biotite only,
# no external DSSP binary required)
# ---------------------------------------------------------------------------

def load_atom_array(structure_path: str, chain_id: Optional[str] = None):
    """Load a PDB file into a biotite AtomArray, filtered to standard amino acids
    (and optionally one chain)."""
    pdb_file = pdb.PDBFile.read(structure_path)
    atom_array = pdb_file.get_structure(model=1)
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    if chain_id:
        mask = atom_array.chain_id == chain_id
        if mask.any():
            atom_array = atom_array[mask]
        # if the requested chain isn't present, fall back to whatever's there
        # (common when structures use a different chain letter than expected)
    return atom_array


def simplify_secondary_structure(sse_code: str) -> str:
    """Collapse biotite's P-SEA 3-state code (a=alpha helix, b=beta sheet,
    c=coil) to the helix/sheet/loop convention used elsewhere in this project."""
    return {"a": "helix", "b": "sheet", "c": "loop"}.get(sse_code, "loop")


def build_residue_table(atom_array) -> list[dict]:
    """
    One row per residue: res_id, res_name, ca_coord, rsa, secondary_structure.
    This is computed once per protein and reused for every mutation on it.
    """
    res_ids_all, res_names_all = struc.get_residues(atom_array)

    # Solvent accessibility: per-atom SASA (Shrake-Rupley), aggregated per residue,
    # normalized by each residue type's theoretical max ASA -> RSA in [0, 1].
    atom_sasa = struc.sasa(atom_array, vdw_radii="Single")
    residue_sasa = struc.apply_residue_wise(atom_array, atom_sasa, np.sum)
    sasa_lookup = dict(zip(res_ids_all, residue_sasa))

    # Secondary structure via P-SEA (backbone-geometry based, no MSA/binary needed).
    try:
        sse = struc.annotate_sse(atom_array)
        sse_lookup = dict(zip(res_ids_all, sse))
    except Exception as e:
        print(f"[WARN] annotate_sse failed ({e}); secondary_structure will be None")
        sse_lookup = {}

    ca_mask = atom_array.atom_name == "CA"
    ca_ids = atom_array.res_id[ca_mask]
    ca_names = atom_array.res_name[ca_mask]
    ca_coords = atom_array.coord[ca_mask]

    table = []
    for res_id, res_name, coord in zip(ca_ids, ca_names, ca_coords):
        res_id = int(res_id)
        sasa = sasa_lookup.get(res_id, np.nan)
        max_asa = MAX_ASA.get(res_name)
        rsa = (sasa / max_asa) if (max_asa and not np.isnan(sasa)) else np.nan
        ss_code = sse_lookup.get(res_id)
        ss = simplify_secondary_structure(ss_code) if ss_code is not None else None
        table.append({
            "res_id": res_id,
            "res_name": res_name,
            "coord": coord,
            "rsa": rsa,
            "ss": ss,
        })
    return table


def compute_position_offset(
    residue_table: list[dict],
    sequence: Optional[str],
    min_match_len: int = 8,
) -> Optional[int]:
    """
    PDB residue numbering (res_id) often does NOT start at 1 — unresolved
    N-terminal residues, cloning/expression tags (e.g. "GSSGSSG..." linkers),
    or crystallographer numbering conventions routinely shift it. Meanwhile,
    mutation `position` columns are typically 1-indexed against the raw
    sequence string, which usually excludes those tags. Mismatched numbering
    silently produces wrong (or all-NaN) features, so this aligns the two.

    Returns an integer offset such that:
        structure_res_id = mutation_position + offset
    or None if no confident alignment could be found (falls back to using
    position directly, with a warning, elsewhere).
    """
    if not sequence or not residue_table:
        return None

    struct_seq = "".join(THREE_TO_ONE.get(r["res_name"], "X") for r in residue_table)
    first_res_id = residue_table[0]["res_id"]

    # Fast path: the structure's observed sequence is an exact contiguous
    # substring of the full given sequence (missing N-terminal residues but
    # no internal gaps in what IS resolved).
    idx = sequence.find(struct_seq)
    if idx != -1:
        return first_res_id - (idx + 1)

    # Fast path 2: just the first stretch of observed residues matches
    # (handles internal gaps later in the chain).
    probe_len = min(10, len(struct_seq))
    if probe_len >= 4:
        probe = struct_seq[:probe_len]
        idx = sequence.find(probe)
        if idx != -1:
            return first_res_id - (idx + 1)

    # Robust fallback: find the longest common contiguous block ANYWHERE in
    # either string. This is what catches cases like a cloning/expression
    # linker (e.g. "GSSGSSG...") prepended in the structure but absent from
    # the given sequence — the real domain match exists later in the chain,
    # not at residue 0.
    import difflib
    matcher = difflib.SequenceMatcher(None, struct_seq, sequence, autojunk=False)
    match = matcher.find_longest_match(0, len(struct_seq), 0, len(sequence))
    if match.size >= min_match_len:
        anchor_res_id = residue_table[match.a]["res_id"]
        return anchor_res_id - (match.b + 1)

    return None


def get_rsa_and_ss(residue_table: list[dict], resnum: int):
    """Look up RSA and 3-state secondary structure for one residue.
    Returns (rsa, ss) or (np.nan, None) if not found."""
    row = next((r for r in residue_table if r["res_id"] == resnum), None)
    if row is None:
        return np.nan, None
    return row["rsa"], row["ss"]


# ---------------------------------------------------------------------------
# Step 3 & 4: contact density and distance to hydrophobic core
# ---------------------------------------------------------------------------

def compute_contact_density(residue_table: list[dict], resnum: int, cutoff: float = CONTACT_CUTOFF_ANGSTROM) -> float:
    """Count CA atoms within `cutoff` Angstroms of the mutated residue's CA
    (excluding itself)."""
    target = next((r for r in residue_table if r["res_id"] == resnum), None)
    if target is None:
        return np.nan
    count = 0
    for r in residue_table:
        if r["res_id"] == resnum:
            continue
        if np.linalg.norm(r["coord"] - target["coord"]) <= cutoff:
            count += 1
    return float(count)


def compute_dist_to_core(
    residue_table: list[dict],
    resnum: int,
    buried_rsa_cutoff: float = BURIED_RSA_CUTOFF,
) -> float:
    """
    Distance from the mutation site's CA to the centroid of buried hydrophobic
    residues (RSA < cutoff and hydrophobic side chain), which approximates the
    hydrophobic core. Falls back to the structure's geometric center if no
    buried-hydrophobic residues are found.
    """
    target = next((r for r in residue_table if r["res_id"] == resnum), None)
    if target is None:
        return np.nan

    core_coords = [
        r["coord"] for r in residue_table
        if r["res_name"] in HYDROPHOBIC_AAS_3LETTER
        and not np.isnan(r["rsa"])
        and r["rsa"] < buried_rsa_cutoff
    ]
    if not core_coords:
        core_coords = [r["coord"] for r in residue_table]  # fallback: geometric center

    centroid = np.mean(np.array(core_coords), axis=0)
    return float(np.linalg.norm(target["coord"] - centroid))


# ---------------------------------------------------------------------------
# Step 5: Conservation score (best-effort, flagged as a limitation)
# ---------------------------------------------------------------------------

def get_conservation_score(protein_id: str, resnum: int, consurf_lookup: Optional[dict] = None) -> float:
    """
    Returns a conservation score if a precomputed lookup (e.g. parsed from
    ConSurf-DB) is provided as {protein_id: {resnum: score}}. Otherwise returns
    NaN. Given the timeline, building fresh MSAs + ConSurf runs per protein is
    deprioritized — this is called out as a limitation in the paper rather than
    blocking the pipeline.
    """
    if consurf_lookup and protein_id in consurf_lookup:
        return consurf_lookup[protein_id].get(resnum, np.nan)
    return np.nan


# ---------------------------------------------------------------------------
# Orchestration: build the full feature table
# ---------------------------------------------------------------------------

@dataclass
class ProteinStructureContext:
    protein_id: str
    structure_path: str
    chain_id: str
    residue_table: list
    position_offset: Optional[int] = None


def prepare_protein_structure(
    protein_id: str,
    out_dir: str,
    pdb_id: Optional[str] = None,
    uniprot_id: Optional[str] = None,
    sequence: Optional[str] = None,
    chain_id: str = "A",
) -> Optional[ProteinStructureContext]:
    """Fetch structure + build its residue table once per protein; reuse across
    all its mutations."""
    structure_path = get_structure_path(protein_id, out_dir, pdb_id, uniprot_id, sequence)
    if structure_path is None:
        print(f"[WARN] No structure found for {protein_id} (tried PDB/AlphaFold/ESMFold)")
        return None
    try:
        atom_array = load_atom_array(structure_path, chain_id=chain_id)
        residue_table = build_residue_table(atom_array)
    except Exception as e:
        print(f"[WARN] Feature extraction failed for {protein_id}: {e}")
        return None

    offset = compute_position_offset(residue_table, sequence)
    if offset is None:
        offset = 0
        if sequence:
            print(f"[WARN] Could not align sequence to structure for {protein_id} — "
                  f"using position as-is (res_id numbering may not match mutation "
                  f"positions; double-check wt_aa against the structure for this protein).")
    elif offset != 0:
        print(f"[INFO] {protein_id}: structure residue numbering offset by {offset:+d} "
              f"relative to mutation positions (auto-detected via sequence alignment).")

    return ProteinStructureContext(protein_id, structure_path, chain_id, residue_table, offset)


def build_structural_feature_table(
    mutations_df: pd.DataFrame,
    structure_dir: str = "structures/",
    consurf_lookup: Optional[dict] = None,
) -> pd.DataFrame:
    """
    mutations_df must have columns: protein_id, position, wt_aa, mut_aa
    and, per protein_id, one of: pdb_id, uniprot_id, or sequence (for structure
    lookup/fold-back). Extra columns are ignored.

    Returns a DataFrame with one row per mutation:
      protein_id, position, wt_aa, mut_aa, rsa, secondary_structure,
      contact_density, dist_to_core, conservation_score
    """
    required = {"protein_id", "position", "wt_aa", "mut_aa"}
    missing = required - set(mutations_df.columns)
    if missing:
        raise ValueError(f"mutations_df is missing required columns: {missing}")

    rows = []
    contexts: dict[str, Optional[ProteinStructureContext]] = {}

    for protein_id, group in mutations_df.groupby("protein_id"):
        if protein_id not in contexts:
            first_row = group.iloc[0]
            contexts[protein_id] = prepare_protein_structure(
                protein_id=protein_id,
                out_dir=structure_dir,
                pdb_id=first_row.get("pdb_id"),
                uniprot_id=first_row.get("uniprot_id"),
                sequence=first_row.get("sequence"),
                chain_id=first_row.get("chain_id", "A"),
            )
        ctx = contexts[protein_id]

        for _, row in group.iterrows():
            position, wt_aa, mut_aa = int(row["position"]), row["wt_aa"], row["mut_aa"]
            if ctx is None:
                rows.append({
                    "protein_id": protein_id, "position": position,
                    "wt_aa": wt_aa, "mut_aa": mut_aa,
                    "rsa": np.nan, "secondary_structure": None,
                    "contact_density": np.nan, "dist_to_core": np.nan,
                    "conservation_score": np.nan,
                })
                continue

            actual_res_id = position + (ctx.position_offset or 0)

            rsa, ss = get_rsa_and_ss(ctx.residue_table, actual_res_id)
            contact_density = compute_contact_density(ctx.residue_table, actual_res_id)
            dist_to_core = compute_dist_to_core(ctx.residue_table, actual_res_id)
            conservation = get_conservation_score(protein_id, position, consurf_lookup)

            rows.append({
                "protein_id": protein_id,
                "position": position,
                "wt_aa": wt_aa,
                "mut_aa": mut_aa,
                "rsa": rsa,
                "secondary_structure": ss,
                "contact_density": contact_density,
                "dist_to_core": dist_to_core,
                "conservation_score": conservation,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# NOTES
# ---------------------------------------------------------------------------
# - No DSSP binary needed: RSA comes from biotite's own Shrake-Rupley SASA
#   calculation, normalized against theoretical max ASA (Tien et al. 2013);
#   secondary structure comes from biotite's P-SEA implementation. Both are
#   pure-Python/compiled-wheel, so `pip install biotite` is all you need —
#   this was changed specifically to avoid the Windows DSSP/WSL headache.
# - Coordinate protein sets with Person A: pull their subsampled protein_id
#   list before running Step 1 here so you fetch structures for exactly the
#   same set.
# - Chain ID: MegaScale designed/natural domains are usually single-chain "A" —
#   verify per structure; multi-chain structures will need the right chain_id
#   in mutations_df.
# - Sanity check early: run this on 3-5 test mutations first (per the Day 1
#   suggested schedule) before scaling to the full subsample.