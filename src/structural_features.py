"""
Person B track - Structural Features pipeline
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

BURIED_RSA_CUTOFF = 0.25
CONTACT_CUTOFF_ANGSTROM = 8.0
HYDROPHOBIC_AAS_3LETTER = {"ALA", "VAL", "LEU", "ILE", "MET", "PHE", "TRP", "TYR", "CYS"}

MAX_ASA = {
    "ALA": 129, "ARG": 274, "ASN": 195, "ASP": 193, "CYS": 167,
    "GLN": 225, "GLU": 223, "GLY": 104, "HIS": 224, "ILE": 197,
    "LEU": 201, "LYS": 236, "MET": 224, "PHE": 240, "PRO": 159,
    "SER": 155, "THR": 172, "TRP": 285, "TYR": 263, "VAL": 174,
}


def fetch_pdb_structure(pdb_id: str, out_dir: str) -> Optional[str]:
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


def load_atom_array(structure_path: str, chain_id: Optional[str] = None):
    pdb_file = pdb.PDBFile.read(structure_path)
    atom_array = pdb_file.get_structure(model=1)
    atom_array = atom_array[struc.filter_amino_acids(atom_array)]
    if chain_id:
        mask = atom_array.chain_id == chain_id
        if mask.any():
            atom_array = atom_array[mask]
    return atom_array


def simplify_secondary_structure(sse_code: str) -> str:
    return {"a": "helix", "b": "sheet", "c": "loop"}.get(sse_code, "loop")


def build_residue_table(atom_array) -> list:
    res_ids_all, res_names_all = struc.get_residues(atom_array)

    atom_sasa = struc.sasa(atom_array, vdw_radii="Single")
    residue_sasa = struc.apply_residue_wise(atom_array, atom_sasa, np.sum)
    sasa_lookup = dict(zip(res_ids_all, residue_sasa))

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


def get_rsa_and_ss(residue_table, resnum: int):
    row = next((r for r in residue_table if r["res_id"] == resnum), None)
    if row is None:
        return np.nan, None
    return row["rsa"], row["ss"]


def compute_contact_density(residue_table, resnum: int, cutoff: float = CONTACT_CUTOFF_ANGSTROM) -> float:
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


def compute_dist_to_core(residue_table, resnum: int, buried_rsa_cutoff: float = BURIED_RSA_CUTOFF) -> float:
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
        core_coords = [r["coord"] for r in residue_table]

    centroid = np.mean(np.array(core_coords), axis=0)
    return float(np.linalg.norm(target["coord"] - centroid))


def get_conservation_score(protein_id: str, resnum: int, consurf_lookup: Optional[dict] = None) -> float:
    if consurf_lookup and protein_id in consurf_lookup:
        return consurf_lookup[protein_id].get(resnum, np.nan)
    return np.nan


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
    required = {"protein_id", "position", "wt_aa", "mut_aa"}
    missing = required - set(mutations_df.columns)
    if missing:
        raise ValueError(f"mutations_df is missing required columns: {missing}")

    rows = []
    contexts = {}

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
