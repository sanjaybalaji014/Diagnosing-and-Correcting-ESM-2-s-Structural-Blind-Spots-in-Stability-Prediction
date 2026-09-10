# III. Methods

## III-A. Dataset

We use the Tsuboyama et al. mega-scale protein stability dataset [CITE],
which measures folding stability (ddG) for a large library of small protein
domains via a high-throughput proteolysis assay: each domain's resistance to
two proteases (trypsin and chymotrypsin) is measured across a titration of
denaturant, yielding an apparent unfolding midpoint (K50) that is converted
to a folding free energy (deltaG) under a two-state folding model, and ddG
is computed as the difference in deltaG between each mutant and its
wild-type reference. The dataset distinguishes natural domains (drawn from
known protein families) from de novo designed domains; we restrict to
natural domains only, since our structural-feature pipeline (III-C) and our
downstream clinical-variant validation (III-F) both concern naturally
occurring, evolutionarily constrained sequences rather than engineered ones.

From the natural-domain subset we sampled 100 small protein domains
(40-72 residues) with experimentally resolved PDB structures, yielding
101,396 unique missense mutations with ground-truth ddG values after
deduplication (`data/processed/person_b_input.csv`). [DISCREPANCY TO
RESOLVE WITH PERSON A: an earlier-stage file in the same pipeline,
`data/processed/esm2_structural_final.csv`, contains 159,426 rows over the
same 100 proteins -- 66 distinct `WT_cluster` groups, all flagged
`is_natural=True` -- and carries the dataset's own train/test partition
(`split` column: 128,540 train rows / 30,886 test rows). We confirmed this
`split` column is protein-level and leak-free (zero protein_id overlap
between its train and test partitions) and that its 20 test proteins are
*identical* to our independently-derived 20-protein holdout
(`test_protein_ids.txt`) -- but its row counts (128,540 / 30,886) do not
match the deduplicated `person_b_input.csv` counts (81,346 / 20,050) for the
same protein sets. The correction model's reported test performance
(`correction_model_results.csv`) uses the 30,886-row count, not the
20,050-row one. Before finalizing this subsection, confirm with Person A
whether the extra ~10,800 test-set rows are legitimate (e.g. multiple
replicate measurements per mutation, retained deliberately) or a
deduplication difference between pipeline stages -- this determines what
"n" should be reported as the true Phase 4 test-set size.]

Proteins were split at the protein level, not the mutation level, into 80
training proteins (81,346 mutations) and 20 held-out test proteins (20,050
mutations; the same 20 proteins used for both the ThermoMPNN structural
baseline comparison in Phase 4 and the correction-model evaluation), so
that no mutation from a test-set protein -- and no near-duplicate domain
from the same `WT_cluster` -- appears anywhere in training. The 20 held-out
test proteins are: 1GL5, 1H8K, 1K1V, 1LP1, 2AMI, 2GP8, 2JZ2, 2KCM, 2L2D,
2LO1, 2LYQ, 2M6Y, 2MCK, 2OCH, 2QFF, 2RJV, 3L1X, 5OAO, 5UP5, 5Z2S.

## III-B. ESM-2 Scoring

For each mutation (wild-type residue *wt*, sequence position *i*, mutant
residue *mt*), we score ESM-2's implied preference for the mutation using
the masked-marginal method: the input sequence is masked at position *i*
only, ESM-2 is run once to obtain the model's marginal distribution over
the twenty amino acids at that position conditioned on the unmasked rest of
the sequence, and the mutation's score is the log-odds ratio between the
mutant and wild-type residue under that single distribution:

    score(wt -> mt @ i) = log P(mt | seq \ {i}) - log P(wt | seq \ {i})

A negative score indicates ESM-2 considers the mutation *less* probable
than the wild type at that position (interpreted, under the model's
zero-shot variant-effect-prediction framing, as destabilizing); a positive
score indicates the reverse. Critically, this requires only one forward
pass per *position*, not per mutation -- the same masked marginal
distribution is reused to score all 19 possible substitutions at that
position -- which is what makes exhaustive or near-exhaustive scoring of a
protein's mutational landscape computationally tractable with a single pass
per residue rather than per (position, substitution) pair.

[UNVERIFIED: the specific ESM-2 checkpoint used (e.g. `esm2_t33_650M_UR50D`,
which would imply 33 transformer layers and a 1280-dimensional hidden
state, consistent with the 33-layer, 1280-dim embeddings referenced in
III-D -- but this is inferred from the probing pipeline's use of "layer 33"
and "1280" as the final-layer dimension, not confirmed directly) is not
recorded anywhere in the committed repo. The scoring script itself was
never checked in: `notebooks/exploration.ipynb` is an empty file, and
`src/esm2_features.py` / `src/data_prep.py` are empty stubs. Confirm the
exact checkpoint name, parameter count, and any batching/truncation
behavior for long sequences with Person A before stating it as fact --
this also bears directly on the `truncated` flag Person A's Phase 5
ESM-2 scoring output carries for 155 of 758 clinical variants (III-F),
which implies a fixed input-length limit was hit for the longer genes.]

## III-C. Structural Feature Extraction

For every mutation we computed four per-residue structural features from
the wild-type structure, following the same pipeline for both the Phase
1-4 natural-domain dataset and the Phase 5 clinical-variant dataset
(III-F), so that features are defined identically across both:

**Relative solvent accessibility (RSA).** Per-atom solvent-accessible
surface area (SASA) is computed via the Shrake-Rupley rolling-sphere
algorithm, summed per residue, and normalized against each residue type's
theoretical maximum ASA (Tien et al. 2013 empirical scale):

    RSA_i = SASA_i / MaxASA(res_name_i)

A residue is classified buried if RSA_i < 0.25, following the common
20-25% convention for the buried/exposed threshold.

**Secondary structure.** Three-state (helix / sheet / loop) secondary
structure is assigned via the P-SEA algorithm, a backbone-geometry-based
method (C-alpha distances and pseudo-dihedral/pseudo-angle patterns) that
requires no external DSSP binary, chosen specifically so the pipeline runs
natively cross-platform via a single `pip install`.

**Contact density.** For a mutated residue at position *i*, contact
density is the count of other residues' C-alpha atoms within an 8
Angstrom radius of residue *i*'s C-alpha:

    contact_density_i = |{ j != i : ||CA_j - CA_i|| <= 8.0 A }|

**Distance to hydrophobic core.** The hydrophobic core is approximated as
the centroid of all buried (RSA < 0.25) hydrophobic residues (Ala, Val,
Leu, Ile, Met, Phe, Trp, Tyr, Cys) in the structure; a mutated residue's
core-distance feature is the Euclidean distance from its C-alpha to that
centroid. Where no such residues exist in a structure (extremely rare),
this falls back to the structure's overall geometric center.

**Structure acquisition and residue-numbering alignment.** Structures were
obtained via a fallback chain: experimental PDB structure first (by PDB
ID), AlphaFold DB predicted structure by UniProt accession if no PDB was
available, and ESMFold (public API) as a last resort for sequences with
neither. Because PDB residue numbering frequently does not start at 1 --
unresolved N-terminal residues, cloning/expression-tag linkers prepended to
the construct -- mutation positions (1-indexed against the raw protein
sequence) were aligned to each structure's actual residue numbering via
longest-common-substring matching between the structure's observed
sequence and the given protein sequence, falling back to a direct
substring search when the structure's sequence is an exact (if truncated)
prefix match. This is the same alignment approach later reused, in a
different context, to resolve the ThermoMPNN PDB-numbering mismatch in
Phase 4 (IV-C) and the AlphaFold-canonical-numbering check in Phase 5
(III-F).

**Coverage.** Across the 100-protein natural-domain set, 97.6% of the
101,396 mutations received valid structural features (non-NaN RSA); 95 of
100 proteins had complete (100%) coverage, and 5 had partial coverage,
reflecting disordered or otherwise unresolved regions in their PDB
structures:

| Protein | Mutations | Valid RSA | Coverage |
|---|---|---|---|
| 2MCH | 1,141 | 323 | 28.3% |
| 1TUD | 1,045 | 248 | 23.7% |
| 2MCK | 946 | 228 | 24.1% |
| 2MWB | 672 | 617 | 91.8% |
| 2QFF | 1,212 | 1,184 | 97.7% |

No protein had zero valid coverage. These partial-coverage proteins were
retained (not excluded) throughout Phases 1-4, with their reduced coverage
noted as a limitation where relevant (notably 2MCK, one of the 20 Phase 4
test proteins, whose ThermoMPNN-matched subset accordingly also shows
reduced coverage relative to the other 19 test proteins -- see IV-C).

## III-D. Mechanistic Probing

To test whether ESM-2's stability-prediction errors reflect structural
information that is genuinely missing from the model's representations, or
structural information that is present but not used when the model scores
mutations, we designed a two-stage diagnostic.

**Stage 1: defining "large-error" mutations.** ESM-2's masked-marginal
score and experimental ddG are on different, incommensurable scales
(log-odds vs. free energy), so a raw difference between them is not a
meaningful notion of "error." We instead fit a single linear calibration
across the entire dataset, ddG ~ esm2_score_masked (ordinary least squares,
one feature), and define each mutation's error as its residual from that
fit:

    predicted_ddG = slope * esm2_score_masked + intercept
    error = ddG_true - predicted_ddG

Mutations in the top 20% by |error| (i.e. above the 80th percentile of
absolute residual) are flagged "large-error" and carried forward to Stage
2; this is the population of mutations ESM-2's calibrated score fits worst,
and therefore the population most informative about *why* the model fails.

**Stage 2: utilization vs. content failure.** For each large-error
mutation, we evaluate whether the true structural properties at that
residue position are linearly decodable from ESM-2's own hidden-state
embedding at that position, using probes (a classifier for buried/exposed
status and for three-state secondary structure, a regressor for contact
density) trained on ESM-2's final-layer (layer 33) embeddings. [UNVERIFIED:
probe classifier/regressor type. Inferred from usage --
`.predict()` returning discrete class labels for `buried` and
`secondary_structure`, continuous values for `contact_density` -- but the
training script for `trained_probes.pkl` is not in this repo (it was
supplied pre-trained by Person A). Confirm exact model class
(LogisticRegression / Ridge, or otherwise), regularization, and training
data (which mutations/positions the probes themselves were fit on, and
whether that overlaps with the Phase 4 test proteins) before stating an
architecture in the paper -- if the probes were trained on positions that
include the Phase 4 test set, that would need to be disclosed as a
methodological caveat.] A large-error mutation is classified:

    utilization_failure  if >= 2 of 3 probes correctly recover the
                          true structural property from ESM-2's embedding
                          at that position
    content_failure      otherwise

For the continuous contact-density target, "correct" is defined relative
to that probe's own typical error: the probe's median absolute residual is
first computed across *all* valid positions in the dataset (not just
large-error ones), and a given mutation's contact-density prediction is
called correct if its residual falls at or below that dataset-wide median.
Intuitively: utilization failure means the structural information was
present and recoverable in ESM-2's representation, but the model's
stability prediction failed to reflect it; content failure means the
information was not recoverable from the embedding at that layer at all.

**Layer sweep.** To characterize *where* in the network structural
information is best represented, and whether that location coincides with
the final layer actually used for stability scoring, the same three probes
were re-evaluated (not restricted to large-error mutations, but across all
5,537 unique (protein, position) pairs with valid structural features) at
seven layers spanning the network's depth:

| Layer | Buried accuracy | Secondary structure accuracy | Contact density R^2 |
|---|---|---|---|
| 5  | 0.850 | 0.957 | 0.548 |
| 10 | 0.871 | 0.956 | 0.575 |
| 15 | 0.904 | 0.968 | 0.688 |
| 20 | 0.929 | 0.969 | 0.716 |
| 25 | 0.946 | **0.971** | 0.735 |
| 30 | **0.954** | 0.968 | 0.752 |
| 33 | 0.919 | 0.957 | **0.757** |

Peak decodability is layer- and feature-dependent, not uniformly earlier
than the final layer as a simplified summary might suggest: buried/exposed
classification peaks at layer 30 (0.954 vs. 0.919 at layer 33, a 3.5-point
drop by the final layer), secondary structure classification peaks at
layer 25 (0.971 vs. 0.957 at layer 33, a 1.4-point drop), while contact
density regression peaks fractionally *at* layer 33 (R^2 = 0.757 vs. 0.752
at layer 30 -- essentially flat, within noise). The two categorical,
more "discrete" structural properties (burial, secondary structure) both
show real, non-trivial degradation by the layer actually used for
stability scoring; the continuous, more local contact-density signal does
not measurably degrade. This distinction is worth preserving in the
Results narrative rather than collapsing to a single "peaks before the
final layer" claim.

## III-E. Correction Model

[SECTION PENDING CONFIRMATION FROM PERSON A -- the following describes only
what can be inferred from the model's *output*, not verified against its
actual training code or saved weights. Do not finalize or cite specific
architecture claims in the submitted paper until confirmed.]

The trained correction model itself (`correction_model_v2.pkl`) was never
committed to the shared repository; only its output predictions
(`data/processed/correction_model_results.csv`) are available for
inspection. That output's columns -- `WT_name`, `mut_type`, `ddG`,
`esm2_score_masked`, `burial`, `secondary_structure`, `chem_transition`,
`corrected_prediction` -- indicate the model takes ESM-2's raw
masked-marginal score together with three structural/chemical features
(burial status, secondary structure category, and a categorical
wild-type-to-mutant "chemical transition" class -- e.g. hydrophobic to
charged, small to bulky; exact category definitions unconfirmed) as input,
and outputs a corrected ddG prediction directly, rather than a residual
correction added on top of the raw ESM-2 score. This "direct prediction"
framing (as opposed to "predict the residual and add it back") is worth
confirming explicitly, since it changes how the model's behavior should be
described and how its gap-closure percentages (IV-C) should be interpreted.

Outstanding items to confirm with Person A before this section is
publication-ready: (1) the exact model class and hyperparameters (the
project's working outline states HistGradientBoostingRegressor, which this
session has not independently verified); (2) the training procedure --
specifically, whether the model was trained only on the 80 training
proteins' mutations, with the 20 held-out test proteins genuinely never
seen during training or feature selection; (3) the precise construction of
the `chem_transition` categorical feature; (4) whether any hyperparameter
tuning used cross-validation that could have touched the test set
indirectly.

## III-F. Clinical Validation Data

To test whether the correction model's improvement over raw ESM-2
generalizes beyond synthetic, in-vitro ddG measurements to a real
downstream clinical task, we assembled an independent variant set spanning
10 disease-relevant genes: APC, BRCA1, BRCA2, MLH1, MSH2, MSH6, PTEN, RB1,
TP53, and VHL. This set combines two variant classes with different
provenance and different implied ground truth:

**Pathogenic variants.** 422 missense variants classified pathogenic in
ClinVar, drawn from two genes only, VHL (167) and PTEN (255) -- chosen for
having a sufficient volume of confidently-classified pathogenic missense
variants in ClinVar, unlike most of the other 8 genes.

**Benign-proxy variants.** 361 missense variants pooled from gnomAD
population allele-frequency data across all 10 genes (common variants,
allele frequency > 0.0001), used as a proxy for "likely benign" under the
standard population-genetics rationale that deleterious missense variants
are subject to negative selection and are systematically depleted from
observed population allele frequencies -- so a variant common enough in a
large population database is, on average, unlikely to be strongly
pathogenic. This is a proxy, not a directly clinically-adjudicated benign
label (unlike the ClinVar pathogenic set), and should be stated as such
wherever the benign class is discussed, including in any AUC/ROC results
in IV-D.

| Gene | Pathogenic (ClinVar) | Benign-proxy (gnomAD) |
|---|---|---|
| VHL | 167 | 4 |
| PTEN | 255 | 2 |
| APC | 0 | 58 |
| BRCA1 | 0 | 64 |
| BRCA2 | 0 | 102 |
| MLH1 | 0 | 23 |
| MSH2 | 0 | 23 |
| MSH6 | 0 | 36 |
| RB1 | 0 | 15 |
| TP53 | 0 | 9 |
| **Total** | **422** | **336** |

(336, not 361: 25 originally-pooled gnomAD benign-proxy variants were
excluded by Person A's own alignment check against each gene's reference
sequence before this session received the scored file; 758 variants total
were carried through ESM-2 scoring.)

**Structural feature extraction.** Structural features for this set were
obtained via AlphaFold DB by UniProt accession -- these genes are large
and mostly multi-domain, and unlike the Phase 1-4 natural-domain set, most
do not have a single experimental PDB structure covering the full
sequence -- using the same RSA / secondary-structure / contact-density /
core-distance pipeline described in III-C, with two extensions specific to
this dataset:

*Multi-fragment structures.* AlphaFold DB splits sequences longer than
2,700 residues into multiple overlapping structure files. We queried
AlphaFold's own prediction API per UniProt accession (rather than assuming
a fixed filename/version pattern) to retrieve every genuine positional
fragment for a given protein, and merged their per-residue tables keyed by
canonical UniProt residue number. This surfaced a related pitfall worth
noting methodologically: the same API also returns alternate splice-isoform
entries under the same accession (confirmed for TP53, which returns 9
total entries under its UniProt accession, only 1 of which is the
canonical full-length monomer used here -- the other 8 correspond to
known shorter TP53 isoforms such as p53beta, p53gamma, and Delta133p53).
Isoform entries are distinguishable from genuine sequence fragments by
their AlphaFold model-entity naming convention (an isoform entry inserts
an extra numeral before the fragment suffix, e.g. `AF-{accession}-9-F1`,
whereas a genuine continuation fragment is named `AF-{accession}-F2`,
`AF-{accession}-F3`); only genuine fragments were used.

*Coverage gap: APC and BRCA2.* Two genes, APC (2,843 residues) and BRCA2
(3,418 residues), are not indexed in AlphaFold DB at all under their
canonical UniProt accessions (P25054 and P51587 respectively) --
confirmed via both AlphaFold's prediction API and a direct structure-file
fetch, both returning HTTP 404. Both exceed AlphaFold DB's practical
proteome-wide length coverage. These two genes (160 of 758 variants,
21.1% of the clinical set) were excluded from structural scoring entirely
and documented as a stated limitation (VI) rather than approximated via a
lower-confidence fallback (e.g. per-domain ESMFold folding), since a
multi-domain 3,400-residue protein folded in disconnected chunks would not
yield a trustworthy global RSA/burial signal.

**Wild-type-residue cross-validation.** As an independent check on
variant-position numbering -- separate from, and performed after, Person
A's own alignment filtering that removed the 25 variants noted above -- we
compared each remaining variant's stated wild-type residue against the
residue AlphaFold's canonical sequence actually has at that position. All
598 variants across the 8 genes with available structures (78.9% of the
758-variant set) matched exactly (100% agreement per gene), corroborating
Person A's filtering independently rather than merely trusting it.

**Structural confidence.** Each variant's structural features are
additionally annotated with AlphaFold's own per-residue confidence score
(pLDDT, extracted directly from the structure file's B-factor field, which
AlphaFold DB repurposes to store this value rather than a true
crystallographic B-factor), binned into AlphaFold's standard four
confidence bands (very low < 50, low 50-70, confident 70-90, very high >
90). 106 of the 598 structurally-scored variants (17.7%) fall in the
low/very-low bands, concentrated in regions AlphaFold itself flags as
likely disordered or poorly-determined, where the buried/exposed framing
underlying our RSA feature is less physically meaningful. These variants
are retained with an explicit `plddt_confidence` flag for downstream
sensitivity analysis (IV-D, VI) rather than silently included or
excluded.
