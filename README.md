# Diagnosing and Correcting ESM-2's Structural Blind Spots in Stability Prediction

Status: Phases 1–4 complete. This file documents provenance decisions that came up
while reconciling Person A's and Person B's independent working sessions, so future
work (and the paper's Methods section) starts from one source of truth instead of
two people's memory of what happened.

## Pipeline summary

1. **Structural features** (`src/run_structural_features.py` → `data/processed/structural_features.csv`) —
   RSA, secondary structure, contact density, distance to hydrophobic core for
   ~100 proteins / 101,396 mutations from the Mega-scale dataset. 97.6% coverage;
   5 proteins partially covered (disordered/unresolved residues), notably `2MCK`.
2. **Utilization vs. content failure classification** (`src/part2_utilization_analysis.py`,
   `src/part3_layerwise_summary.py`) — per-layer linear/logistic probes on ESM-2
   hidden states test whether structural properties are linearly decodable from the
   embeddings at each layer. Peak decodability is at layers 25–30, not layer 33
   (the layer actually used for stability scoring).
3. **Cluster cross-referencing** (`src/part3_cluster_utilization.py`) — joins the
   utilization/content verdict to Person A's burial × secondary-structure ×
   chemical-transition clusters (`data/processed/cluster_utilization_summary.csv`,
   `headline_finding_breakdown.csv`).
4. **Structure-aware baseline comparison** (`src/compare_baselines.py`) — raw ESM-2
   vs. the correction model vs. ThermoMPNN, on the 20 held-out test proteins
   (`data/processed/baseline_comparison_summary.csv`).

## Resolved discrepancies (Person A / Person B session reconciliation)

### 1. Headline finding: magnitude vs. composition

Person A's t-test (buried+charge-changing vs. other buried mutations, t=32.3) and
Person B's utilization-failure breakdown are **compatible, not competing** — they
answer different questions:

- Person A's test: does *mean error* differ between the groups? Yes, by close to a
  full kcal/mol in every secondary-structure fold (see `headline_finding_breakdown.csv`).
- Person B's analysis: does the *reason* for the error differ — i.e. is structural
  information still linearly decodable from ESM-2's embeddings at these positions?
  No — utilization-failure rate is ~83–95% in every fold, headline cluster and
  otherwise, moving by at most ~2 points.

**Joint claim for the paper:** buried, charge-changing mutations expose ESM-2's
structural blind spot *more severely*, not *differently* — the failure mode
(ignoring decodable structural information) is uniform across clusters; only the
resulting error magnitude varies with mutation type.

### 2. Canonical correction model

There is one correction model: Person A's, saved as `correction_model_v2.pkl` /
`correction_model_results.csv`. `compare_baselines.py` reads the `corrected_prediction`
column directly from that CSV — it does not retrain or refit anything.

The small numeric gap some earlier summaries showed (Spearman 0.612 vs. 0.610, MAE
0.600 vs. 0.601) is a row-count artifact, not a second model: `compare_baselines.py`
scores only the 30,726 / 30,886 rows (99.5%) that matched to a ThermoMPNN prediction
after per-protein PDB-numbering-offset correction; the small remainder is almost
entirely `2MCK`, already flagged as partially covered in Phase 1. For the three-way
comparison, use the 30,726-row numbers (all three methods scored on identical rows);
Person A's full-30,886-row number is the correct one to cite for the correction
model's standalone test performance.

### 3. Structure-aware baseline choice

ThermoMPNN was chosen over RaSP (the originally discussed first choice) after RaSP's
repo was found to be pinned to Python 3.6 / PyTorch 1.2.0 / CUDA 10.0 with a dead
Colab notebook — a dependency-risk decision made before any install was attempted,
not a performance comparison. ELASPIC-2 was never evaluated; once ThermoMPNN was
confirmed runnable on a modern stack (Python 3.10, trained on the same Mega-scale
data this project already uses) there was no need to fall back to the third option.
State it in Methods as: *chosen for environment compatibility and shared training
data, not benchmarked against ELASPIC-2.*

## Open / unresolved

### 4. Phase 5 scope

**Not resolved.** No file matching a ClinVar/gnomAD clinical variant validation
pipeline (e.g. `phase5_clinvar_gnomad_variants.csv`) exists anywhere in this repo,
and neither `README.md` nor `notebooks/exploration.ipynb` recorded the original
master plan's Phase 5 text. Do not assume either description (paper write-up, or
clinical variant validation) is correct until the original master plan text is
re-confirmed directly with Person A/Sanjay.
