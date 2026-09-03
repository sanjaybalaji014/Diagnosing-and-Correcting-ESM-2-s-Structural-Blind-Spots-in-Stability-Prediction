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
    return ProteinStructureContext(protein_id, structure_path, chain_id, residue_table)
 
 
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
 
            rsa, ss = get_rsa_and_ss(ctx.residue_table, position)
            contact_density = compute_contact_density(ctx.residue_table, position)
            dist_to_core = compute_dist_to_core(ctx.residue_table, position)
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