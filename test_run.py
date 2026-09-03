import sys
sys.path.insert(0, "src")
import pandas as pd
from structural_features import build_structural_feature_table

# a real, well-known small protein (ubiquitin) as a smoke test
mutations_df = pd.DataFrame([
    {"protein_id": "1UBQ", "position": 10, "wt_aa": "L", "mut_aa": "A",
     "pdb_id": "1UBQ", "uniprot_id": None, "sequence": None},
    {"protein_id": "1UBQ", "position": 30, "wt_aa": "I", "mut_aa": "V",
     "pdb_id": "1UBQ", "uniprot_id": None, "sequence": None},
])

result = build_structural_feature_table(mutations_df, structure_dir="structures/")
print(result)