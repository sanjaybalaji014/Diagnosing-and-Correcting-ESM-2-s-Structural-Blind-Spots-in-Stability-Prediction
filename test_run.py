import sys
sys.path.insert(0, "src")
import pandas as pd
from structural_features import build_structural_feature_table

mutations_df = pd.read_csv("data/raw/person_b_input.csv")
print("Total mutations:", len(mutations_df))
print("Unique proteins:", mutations_df["protein_id"].nunique())

test_df = mutations_df.head(5)
result = build_structural_feature_table(test_df, structure_dir="structures/")
print(result)
