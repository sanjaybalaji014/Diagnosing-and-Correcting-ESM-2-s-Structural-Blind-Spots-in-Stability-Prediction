# Paper Outline: Diagnosing and Correcting ESM-2's Structural Blind Spots in Mutation Stability Prediction

Status legend: [DONE] = verified to exist in repo/committed; [PENDING] = not yet built; [UNVERIFIED] = claimed but not independently confirmed by this session.

## Title
"Diagnosing and Correcting ESM-2's Structural Blind Spots in Mutation Stability Prediction" [DONE -- set]

## Abstract (write last, ~200 words)
One sentence each: motivation -> method -> Phase 3 finding -> Phase 2 mechanism -> Phase 4 result -> Phase 5 validation -> one-sentence takeaway.

## I. Introduction
- Para 1: pLMs (ESM-2) widely used for stability/pathogenicity prediction; usage assumes uniform reliability.
- Para 2: gap -- no systematic characterization of when/why pLMs fail, or whether that's a content or utilization problem.
- Para 3: contributions, numbered list (taxonomy, mechanistic probing, correction model, clinical validation).
- No graphics.

## II. Related Work
- RaSP, ELASPIC-2, ESMTherm/SaProtDG (existing structure+sequence hybrids)
- ThermoMPNN (baseline) [DONE -- used in Phase 4]
- Prior pLM interpretability/probing work
- No graphics; one short paragraph each.

## III. Methods
- III-A. Dataset: Tsuboyama mega-scale, natural domain filtering (450->100 sampled), protein-level 80/20 train/test split. Summary stats table only.
- III-B. ESM-2 Scoring: masked-marginal method, formula. No graphic.
- III-C. Structural Feature Extraction: RSA, secondary structure, contact density, dist-to-core; PDB->AlphaFold->ESMFold fallback; 97.6% coverage (Phase 1). [DONE]
- III-D. Mechanistic Probing: layers (5,10,15,20,25,30,33), logistic/ridge probes, utilization vs. content failure definition. [DONE -- layerwise_probe_performance.csv]
- III-E. Correction Model: [UNVERIFIED -- claimed HistGradientBoostingRegressor; actual .pkl not in repo, only its output CSV. Confirm with Person A directly before stating architecture as fact.]
- III-F. Clinical Validation Data: ClinVar VHL/PTEN pathogenic (422) + pooled gnomAD benign-proxy across 10 genes (361 -> 336 after Person A's alignment QC, 758 total scored by ESM-2).
  [CORRECTION NEEDED: usable-with-structural-features count is 598/758 (78.9%), not 758/758. 160 variants (APC + BRCA2, 58+102) have zero AlphaFold DB coverage -- confirmed via direct API/file 404s, not an alignment issue. Separately, 22 gnomAD benign variants were dropped by Person A's own alignment check before this session ever touched the data.]

## IV. Results
- IV-A. Failure-Mode Taxonomy (Phase 3): stratified error table + bar/heatmap (burial x chemistry transition) + t-test (t=32.3, p~5.4e-222). [DONE -- data exists, chart pending]
- IV-B. Mechanistic Probing (Phase 2): layer-vs-accuracy line plot; utilization-failure rate ~85-95% flat across clusters; joint claim (magnitude varies, failure-mode composition doesn't). [DONE -- data exists, chart pending]
- IV-C. Correction Model Performance (Phase 4): three-way comparison table (Spearman/Pearson/MAE), predicted-vs-actual scatter (2 panels), gap-closure % (43.5/65.3/96.5), feature importance.
  [DONE: comparison table (baseline_comparison_summary.csv). PENDING: scatter plots, feature importance (not found in repo -- confirm this exists before citing it).]
- IV-D. Clinical Variant Separation (Phase 5): AUC table (full/high-confidence/buried+charge-changing subsets), ROC curves (raw vs corrected), n=3 benign caveat stated in text.
  [PENDING -- correction model hasn't been run on Phase 5 structural features yet; no AUC/ROC numbers exist yet. This is the next concrete step, blocked on Person A.]

## V. Discussion
- Synthesize IV-A-IV-D into one causal narrative (taxonomy -> mechanism -> correction -> clinical payoff).
- Explain "ceiling effect" for aggregate Phase 5 AUC (conservation signal dominates) -- pending Phase 5 results to know if this actually holds.
- ThermoMPNN vs RaSP rationale (briefly, in Methods or here). [DONE -- documented in README discrepancy #3]

## VI. Limitations
- gnomAD-conservation confound
- Small training domain sizes (40-72 aa) vs. large clinical genes
- APC/BRCA2 excluded (no AlphaFold coverage) [DONE -- confirmed, documented]
- Low-pLDDT variants: 106/598 (17.7%), not 17.7% of the full set -- correct this figure when writing (sensitivity-checked via plddt_confidence column, not dropped) [DONE -- data exists]
- Isoform/alignment mismatches: 22/758 gnomAD benign rows dropped (not 25/783 -- that 25 count was before Person A's own scored file; verify final number against phase5_esm2_scored.csv, which already reflects the drop)
- Small benign-proxy sample size in hard-cluster subset

## VII. Conclusion
3-4 sentences, no new information.

## Acknowledgments
AI disclosure statement (per IEEE policy), funding/support notes.

## References
IEEE format, .bib file. Needed: ESM-2 paper, Tsuboyama et al. mega-scale dataset paper, RaSP, ELASPIC-2, ThermoMPNN, ClinVar, gnomAD.
