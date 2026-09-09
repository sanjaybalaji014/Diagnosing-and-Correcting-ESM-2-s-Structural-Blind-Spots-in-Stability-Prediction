"""
Diagnose why a given PDB structure produces zero valid RSA values.

Usage:
    python src/diagnose_structure.py structures/1UFM.pdb
"""
import sys
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

path = sys.argv[1] if len(sys.argv) > 1 else "structures/1UFM.pdb"

pdb_file = pdb.PDBFile.read(path)

print(f"--- {path} ---")
print("models in file:", pdb_file.get_model_count())

atom_array = pdb_file.get_structure(model=1)
print("total atoms (model 1):", len(atom_array))
print("chains present:", sorted(set(atom_array.chain_id)))
print("unique res_names (first 20):", sorted(set(atom_array.res_name))[:20])

filtered = atom_array[struc.filter_amino_acids(atom_array)]
print("atoms after filter_amino_acids:", len(filtered))

if len(filtered) == 0:
    print("\n>>> filter_amino_acids removed EVERYTHING — that's the bug. "
          "res_names above are probably non-standard (modified residues, "
          "or this file uses HETATM records) so biotite doesn't recognize them as amino acids.")
    sys.exit(0)

res_ids, res_names = struc.get_residues(filtered)
print("residue count after filter:", len(res_ids))

try:
    atom_sasa = struc.sasa(filtered, vdw_radii="Single")
    print("sasa computed OK, array len:", len(atom_sasa))
    import numpy as np
    n_nan = np.isnan(atom_sasa).sum()
    print("NaN values in atom_sasa:", n_nan, "/", len(atom_sasa))
except Exception as e:
    print("\n>>> struc.sasa() RAISED AN EXCEPTION:")
    print(type(e).__name__, str(e))