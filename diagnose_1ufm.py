import sys
sys.path.insert(0, "src")
import pandas as pd
from structural_features import get_structure_path, load_atom_array, build_residue_table
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

# get the sequence Person A gave for this protein
mutations_df = pd.read_csv("data/raw/person_b_input.csv")
row = mutations_df[mutations_df["protein_id"] == "1UFM.pdb"].iloc[0]
sequence = row["sequence"]
print("Given sequence:", sequence)
print("Given sequence length:", len(sequence))

path = get_structure_path("1UFM.pdb", "structures/", pdb_id="1UFM")
print("structure path:", path)

# check what chains actually exist in the raw file, before any filtering
pdb_file = pdb.PDBFile.read(path)
raw_array = pdb_file.get_structure(model=1)
print("all chain IDs present:", sorted(set(raw_array.chain_id)))
print("all res_names present (unique):", sorted(set(raw_array.res_name)))

atom_array = load_atom_array(path, chain_id="A")
print("n atoms after chain A + amino-acid filter:", len(atom_array))

table = build_residue_table(atom_array)
print("n residues in table:", len(table))
if table:
    print("first 10 residues:", [(r["res_id"], r["res_name"]) for r in table[:10]])
else:
    print("EMPTY residue table")
