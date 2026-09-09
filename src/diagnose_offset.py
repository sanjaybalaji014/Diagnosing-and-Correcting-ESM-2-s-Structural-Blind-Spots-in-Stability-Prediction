"""
Diagnose why position-offset alignment fails for a given protein.

Usage:
    python src/diagnose_offset.py 1UFM.pdb
    (protein_id must match a value in data/processed/person_b_input.csv)
"""
import sys
sys.path.insert(0, "src")

import pandas as pd
from structural_features import (
    load_atom_array, build_residue_table, compute_position_offset, THREE_TO_ONE,
)

protein_id = sys.argv[1] if len(sys.argv) > 1 else "1UFM.pdb"

mutations_df = pd.read_csv("data/processed/person_b_input.csv")
group = mutations_df[mutations_df["protein_id"] == protein_id]
if group.empty:
    print(f"No rows found for protein_id={protein_id!r} in person_b_input.csv")
    sys.exit(1)

first_row = group.iloc[0]
pdb_id = first_row.get("pdb_id")
sequence = first_row.get("sequence")
print("pdb_id:", pdb_id)
print("sequence (from CSV):", sequence)
print("sequence length:", len(sequence) if isinstance(sequence, str) else None)
print("position range in mutations:", group["position"].min(), "-", group["position"].max())

structure_path = f"structures/{pdb_id}.pdb"
atom_array = load_atom_array(structure_path, chain_id="A")
residue_table = build_residue_table(atom_array)

struct_seq = "".join(THREE_TO_ONE.get(r["res_name"], "X") for r in residue_table)
print("\nstruct_seq (from structure):", struct_seq)
print("struct_seq length:", len(struct_seq))
print("res_id range in structure:", residue_table[0]["res_id"], "-", residue_table[-1]["res_id"])

offset = compute_position_offset(residue_table, sequence)
print("\ncomputed offset:", offset)

if offset is None:
    print("\n>>> Offset detection FAILED to find a confident match.")
    print(">>> This means every mutation position falls back to using the raw")
    print(">>> position number as the structure res_id, which will miss every")
    print(">>> residue if the true res_id range (above) doesn't start near 1.")