import sys
sys.path.insert(0, "src")
import pandas as pd
from structural_features import build_structural_feature_table

mutations_df = pd.read_csv("data/raw/person_b_input.csv")
print("Total mutations:", len(mutations_df))
print("Unique proteins:", mutations_df["protein_id"].nunique())

feat_df = build_structural_feature_table(mutations_df, structure_dir="structures/")
feat_df.to_csv("data/processed/structural_features.csv", index=False)

print("Saved", len(feat_df), "rows to data/processed/structural_features.csv")
print(feat_df.head(10))
print("NaN counts:")
print(feat_df[["rsa", "secondary_structure", "contact_density", "dist_to_core"]].isna().sum())
