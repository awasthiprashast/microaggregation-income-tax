<!--
  docs/REPORT.md is GENERATED from docs/report_template.md by
  `python run_experiments.py`. Double-brace placeholders are filled with computed results
  (see microagg/report_builder.py).
-->

# Micro-aggregation in Income and Tax Data Publication – Implementation

**Project report** · Data Privacy · synthetic dataset of {{n_records}} taxpayers · random seed {{seed}}

> Every number in this report was produced by `run_experiments.py` (Python {{python}},
> NumPy {{numpy}}, pandas {{pandas}}) and inserted automatically. Re-running the script with
> the same seed reproduces them exactly; a different seed or dataset regenerates the report.

---

## Abstract

Tax administrations and statistical offices are asked to publish income and tax microdata for
research and policy analysis, but such records are among the most sensitive personal data. This
project implements **micro-aggregation** with the **MDAV** (Maximum Distance to Average Vector)
heuristic as a statistical disclosure control method for an income/tax dataset, together with an
end-to-end pipeline: synthetic data generation, validation and imputation, anonymisation, a
release guard, utility and disclosure-risk evaluation, attack simulations and an interactive
Streamlit dashboard. On {{n_records}} synthetic taxpayers, MDAV with k = 3, 5, 10, 20 yields
k-anonymous releases whose record-linkage success falls from {{dbrl_all_%|mdav|3}} % to
{{dbrl_all_%|mdav|20}} % while information loss (log scale) rises from {{il_log_percent|mdav|3}} %
to {{il_log_percent|mdav|20}} % and means and totals (e.g. total tax revenue) are preserved exactly.
The experiments also show the limits of the method: even at k = 20, {{interval_disclosure_gross_income_%|mdav|20}} %
of taxpayers have their published income within ±10 % of the truth, and an attacker who knows
the incomes of 90 % of the population recovers {{collusion_f0.9_%|mdav|20}} % of the remaining
incomes exactly. Micro-aggregation substantially reduces, but does not eliminate, disclosure risk.

## 1. Introduction

Income-tax microdata – one record per taxpayer with demographics, income components, deductions
and tax – are invaluable for evaluating tax reforms, measuring inequality and forecasting revenue.
Releasing them as-is, even without names, is unsafe: a combination of age, region, occupation and
family situation is often unique, and anyone who can single a person out learns their income and
tax. Statistical disclosure control (SDC) offers *masking* methods that perturb the data so that
individuals are protected while aggregate statistics remain useful.

Micro-aggregation is a masking method designed for continuous data. It groups similar records
into clusters of at least *k* records and publishes the cluster averages instead of the individual
values. It is used by statistical agencies (it is implemented in μ-ARGUS and sdcMicro) and,
with a joint partition, it yields *k*-anonymity for numerical data (Domingo-Ferrer & Torra, 2005).

## 2. Problem statement

Given a microdata file of taxpayers with direct identifiers, quasi-identifiers (QIs) and
confidential financial attributes, produce a releasable file such that

1. no released record can be linked to a specific taxpayer with probability above 1/k,
2. original confidential values are not exposed – neither in the released file nor in the tools
   used to produce it,
3. the released file supports the analyses tax researchers actually run (means, totals,
   distributions, inequality, effective tax rates, regression), and
4. the residual disclosure risk is measured and reported honestly.

## 3. Objectives

* Implement MDAV micro-aggregation, cleanly and reproducibly, with documented complexity.
* Build a realistic synthetic income/tax population with correlated attributes and heavy tails.
* Validate input data and handle missing and invalid values.
* Quantify utility (information loss, distributional and analytical validity) and risk
  (identity and attribute disclosure, attack simulations) for k = 3, 5, 10, 20.
* Compare MDAV with univariate (individual ranking) and random-partition baselines.
* Provide an interactive dashboard and an export path that cannot leak original values.
* Discuss limitations and attacks explicitly.

## 4. Literature and conceptual background

**Statistical disclosure control.** SDC distinguishes *identity disclosure* (a released record is
linked to a person) from *attribute disclosure* (something is learnt about a person, whether or
not their record is found) (Hundepool et al., 2012). Masking methods are either
non-perturbative (suppression, generalisation) or perturbative (noise, swapping,
micro-aggregation).

**k-anonymity** (Sweeney, 2002) requires every combination of QI values to be shared by at least
*k* records. It bounds identity disclosure but is vulnerable to the *homogeneity* and
*background-knowledge* attacks, which motivated **l-diversity** (Machanavajjhala et al., 2007) and
**t-closeness** (Li et al., 2007).

**Micro-aggregation.** Defay & Nanopoulos (1992) introduced fixed-size micro-aggregation;
Domingo-Ferrer & Mateo-Sanz (2002) formalised it as minimising the within-group sum of squares
subject to a minimum group size, showed that optimal groups have between k and 2k−1 records, and
proposed MDAV. Optimal multivariate micro-aggregation is NP-hard (Oganian & Domingo-Ferrer, 2001);
the univariate case has a polynomial optimum (Hansen & Mukherjee, 2003). Domingo-Ferrer & Torra
(2005) showed that micro-aggregating all QIs with a joint partition achieves k-anonymity for
continuous data. Variants include V-MDAV (variable group sizes) and MD-MHM.

**Tabular confidentiality rules.** Tax agencies protect published cells with the
*(n, k)-dominance* and *p %-rule*, which flag cells where one or two contributors dominate the
total; the same logic applies to a micro-aggregated group mean.

**Differential privacy** (Dwork, 2006) offers a formal guarantee against arbitrary auxiliary
information, at the price of calibrated noise; micro-aggregation offers no such guarantee.

## 5. Methodology

```
raw CSV ──► validation & cleaning ──► distance-space encoding ──► partition (MDAV | IR | random)
                                                                          │
           release guard ◄── shuffle & drop ids ◄── aggregation (mean / mode, one partition)
                 │
                 └──► released CSV + metadata          utility & risk metrics ◄── (internal only)
```

1. **Data.** A synthetic population generated with a fixed seed (Section 9).
2. **Validation and cleaning** (`preprocessing.py`): schema check, label normalisation,
   de-duplication, numeric coercion, domain ranges, edit rules (deductions ≤ gross income),
   *deductive repair* with the identity taxable = gross − deductions, removal of mostly-empty
   rows, median imputation within occupation (numeric) and mode imputation (categorical),
   recomputation of tax when missing.
3. **Encoding** (`features.py`): z-scores, log1p for monetary attributes, weighted one-hot
   encoding for categories (weight 1/√2).
4. **Partitioning** with MDAV; baselines: individual ranking and random partition.
5. **Aggregation** with a single partition for all attributes: mean (numeric), mode (categorical).
6. **Release** (`release.py`): whitelist of aggregated attributes, rounding, seeded shuffle, fresh
   `release_id`, and blocking checks (k-anonymity, completeness, id hygiene).
7. **Evaluation** (`utility.py`, `privacy.py`) on the internal, row-aligned original/anonymised
   pair – never exported.

## 6. The MDAV algorithm

MDAV repeatedly takes the record *r* farthest from the centroid of the unassigned records and
the record *s* farthest from *r*, and forms a group from each of them with their k−1 nearest
neighbours, until fewer than 3k records remain; then it forms one or two final groups so that
every group has between k and 2k−1 records. Each record is replaced by its group centroid.

*Intuition.* Extreme records are the hardest to hide, so they are grouped first, while they still
have the largest choice of neighbours; *r* and *s* lie at opposite ends of the cloud, so the two
groups formed in each round do not compete for the same neighbours.

*Complexity.* An iteration over *m* remaining records computes three distance vectors
(O(md)) and two linear-time k-selections, and removes 2k records. Over n/(2k) iterations:

T(n) = Σᵢ O(d·(n − 2ki)) ≈ O(n²·d / 4k) = **O(n² d / k)** time, **O(n d)** memory.

Measured on {{n_features}}-dimensional encodings, the log–log slope of runtime is
**{{slope_n|.2f}}** in *n* and **{{slope_k|.2f}}** in *k*, matching the predicted exponents 2 and −1.

{{table:scalability}}

![Runtime](../results/figures/runtime.png)

Full derivations, a worked example and proofs are in [`docs/ALGORITHM.md`](ALGORITHM.md).

## 7. Implementation

| Module | Responsibility |
|---|---|
| `microagg/config.py` | attribute roles, valid ranges, `AnonymizationConfig` |
| `microagg/tax.py` | stylised progressive tax schedule (slabs, rebate, surcharge, cess) |
| `microagg/data_generator.py` | reproducible synthetic population + injected dirt |
| `microagg/preprocessing.py` | validation, edit rules, deductive repair, imputation |
| `microagg/features.py` | distance-space encoder |
| `microagg/mdav.py` | MDAV, individual ranking, random partition, aggregation |
| `microagg/anonymizer.py` | pipeline returning an internal `AnonymizationResult` |
| `microagg/release.py` | release builder and release guard |
| `microagg/utility.py`, `privacy.py` | utility metrics, risk metrics, attacks |
| `microagg/safe_display.py` | banding / safe summaries for anything shown on screen |
| `microagg/experiments.py`, `plotting.py`, `report_builder.py` | experiments, figures, this report |
| `app.py` | Streamlit dashboard |
| `tests/test_microagg.py` | unit tests (group-size invariants, determinism, k-anonymity, guard, validation) |

**Safeguards against accidental exposure.** (i) The release is built from a *whitelist* of
aggregated attributes; `taxpayer_id` and the derived `tax_bracket` can never be exported.
(ii) Rows are shuffled with a seeded permutation and re-numbered, so row order cannot be joined
back to the source. (iii) The release guard blocks any file that is not k-anonymous – in the
experiments it blocked: *{{guard_blocked}}*. (iv) The dashboard shows original monetary values only
as bands, never shows per-record errors (released value + error = original), never shows the
minimum or maximum of an original attribute, trims distribution plots to P1–P99, and keeps exact
originals behind an explicit, off-by-default *custodian view*. (v) All randomness is seeded.

## 8. Experimental setup

* **Data:** {{n_raw_rows}} raw rows → {{n_records}} clean records after removing
  {{duplicates_removed}} duplicates; {{cells_repaired}} cells repaired exactly through the
  accounting identity and {{cells_imputed}} cells imputed; {{rows_dropped}} rows dropped.
* **Released attributes:** age, dependents, gross income, deductions, taxable income, capital
  gains, tax paid (numeric); gender, marital status, education, occupation, region, city tier
  (categorical). All are micro-aggregated with one partition.
* **Methods:** MDAV; individual ranking (IR); random partition. **k ∈ { {{k_values}} }.**
* **Attacker models:** (a) worst case – the attacker holds the original records (DBRL on all
  attributes); (b) the attacker knows the QIs only; (c) the attacker knows the true incomes of a
  share *f* of the population (collusion).
* **Validation of cleaning:**

{{table:validation_report}}

## 9. Dataset

The synthetic population mimics the structure of personal income-tax returns: age-dependent
marital status and retirement, dependents driven by family status, education- and
occupation-dependent log-normal incomes with an experience curve and a city-tier premium,
a Pareto tail for business owners and professionals, deductions whose take-up grows with
income, capital gains for a minority, and tax computed with a stylised progressive schedule
(rebate up to ₹7 L, slabs 0–30 %, surcharge, 4 % cess; *illustrative only, not tax law*).
About 3 % of cells are deliberately made missing and ~0.4 % invalid, and a few rows are duplicated.

## 10. Results

### 10.1 Utility

{{table:utility}}

*IL raw* is the classic 100·SSE/SST on z-scored values; *IL log* uses log1p for money;
*Var. ratio* is the mean released/original variance ratio; *CI overlap* is the mean 95 %
confidence-interval overlap of the regression coefficients.

Income and tax statistics on the MDAV releases:

{{table:income_stats}}

![Information loss](../results/figures/information_loss.png)

![Income distribution](../results/figures/income_distribution.png)

![Effective tax rate by quintile](../results/figures/quintile_tax.png)

Per-attribute statistics, MDAV k = {{k_detail}}:

{{table:attribute_mdav_k5}}

Regression of log(1 + gross income) on demographics, MDAV k = {{k_detail}}:

{{table:regression_mdav_k5}}

### 10.2 Privacy

{{table:privacy}}

*Min. class* = smallest equivalence class over all released attributes; *DBRL* = expected success
of distance-based record linkage; *Income ±10 %* = interval disclosure of gross income;
*QI inference* = success of the QI-based income-inference attack (baseline without the release:
{{baseline_success_%|mdav|5}} %); *Bracket homog.* = records in groups whose members all share
one tax bracket; *Mean l* = mean distinct tax brackets per group; *p %-rule* = records in groups
failing the p %-rule (p = 10); *Collusion f = 0.5* = targets recovered exactly when the attacker
knows half of all incomes.

![Linkage risk](../results/figures/linkage_risk.png)

![Attribute disclosure](../results/figures/attribute_disclosure.png)

![Collusion attack](../results/figures/collusion.png)

### 10.3 Comparison of original and anonymised records

Eight random records, MDAV k = {{k_detail}}. In line with the project's exposure rules, original
monetary values are shown as bands and ages as decades; per-record errors are deliberately not
shown.

{{table:record_comparison_mdav_k5}}

### 10.4 Ablation: design of the distance space (MDAV, k = {{k_detail}})

{{table:ablation}}

## 11. Privacy analysis

* **Identity disclosure is bounded as promised.** Every MDAV and random-partition release is
  k-anonymous (minimum class = k), and no record's linkage probability exceeds 1/k
  (max. per-record risk {{dbrl_all_max_record_risk|mdav|3|.3f}} at k = 3,
  {{dbrl_all_max_record_risk|mdav|20|.3f}} at k = 20). Even for the worst-case attacker holding the
  original file, expected linkage success falls from {{dbrl_all_%|mdav|3}} % (k = 3) to
  {{dbrl_all_%|mdav|20}} % (k = 20); with QIs only, from {{dbrl_qi_%|mdav|3}} % to {{dbrl_qi_%|mdav|20}} %.
  Individual ranking, in contrast, leaves records essentially unique: linkage succeeds for
  {{dbrl_all_%|individual_ranking|3}} % of records at k = 3 and still {{dbrl_all_%|individual_ranking|20}} %
  at k = 20, which is why the release guard blocks it.
* **Attribute disclosure is NOT bounded by k.** Because MDAV groups similar taxpayers, the
  published income is within ±10 % of the true income for {{interval_disclosure_gross_income_%|mdav|3}} %
  of taxpayers at k = 3 and still {{interval_disclosure_gross_income_%|mdav|20}} % at k = 20 (random
  grouping: {{interval_disclosure_gross_income_%|random|20}} %). An attacker who only knows a target's
  QIs estimates the income within ±10 % for {{inference_success_%|mdav|3}} % of targets at k = 3
  versus a {{baseline_success_%|mdav|3}} % baseline without the release; the gain shrinks to
  {{disclosure_gain_pp|mdav|20}} percentage points at k = 20.
* **Homogeneity.** {{homogeneous_records_%|mdav|3}} % of records (k = 3) and {{homogeneous_records_%|mdav|20}} %
  (k = 20) sit in groups whose members all share one tax bracket – an attacker who identifies the
  group learns the bracket with certainty. The minimum distinct l-diversity is
  {{min_l_diversity|mdav|20|.0f}} even at k = 20, and the worst group's t-closeness distance is
  {{max_t_closeness|mdav|20|.2f}}.
* **Dominance.** Groups failing the (1, 50 %) dominance rule: {{dominance_groups_%|mdav|3|.2f}} % at
  k = 3, {{dominance_groups_%|mdav|5|.2f}} % at k = 5; p %-rule failures: {{p_rule_groups_%|mdav|3|.2f}} %.
  MDAV's similarity grouping keeps very high earners together, so a single outlier rarely
  dominates; random grouping is far worse ({{dominance_groups_%|random|3}} % of groups at k = 3).
* **Background knowledge.** If the attacker already knows half of all incomes, they recover
  {{collusion_f0.5_%|mdav|3}} % of the other incomes exactly at k = 3, {{collusion_f0.5_%|mdav|5}} % at
  k = 5 and {{collusion_f0.5_%|mdav|10|.2f}} % at k = 10. With 90 % known, the figures are
  {{collusion_f0.9_%|mdav|3}} %, {{collusion_f0.9_%|mdav|5}} %, {{collusion_f0.9_%|mdav|10}} % and
  {{collusion_f0.9_%|mdav|20}} % for k = 3, 5, 10, 20. Simulations match the f^(|G|−1) prediction.

## 12. Utility analysis

* **First moments are exact.** Means of every attribute and the total tax revenue are preserved
  for every k (largest relative deviation of any attribute mean at k = 20:
  {{max_abs_mean_diff_%|mdav|20|.6f}} %).
* **Information loss grows with k.** For MDAV, IL raw rises from {{il_percent|mdav|3}} % (k = 3) to
  {{il_percent|mdav|20}} % (k = 20) and IL log from {{il_log_percent|mdav|3}} % to {{il_log_percent|mdav|20}} %.
  The raw figure looks high because incomes are heavy-tailed: the top 1 % of earners account for
  {{top1_share_of_sst|.1f}} % of the total sum of squares of gross income, so raw IL mainly measures
  how well a handful of very rich taxpayers are grouped. Random grouping, with the same
  k-anonymity, loses {{il_percent|random|3}}–{{il_percent|random|20}} % – MDAV's similarity search is
  what makes the release useful.
* **Distributions and inequality.** The Gini coefficient of gross income moves from
  {{gini_gross_income_original|mdav|3|.3f}} to {{gini_gross_income_anonymized|mdav|3|.3f}} (k = 3) and
  {{gini_gross_income_anonymized|mdav|20|.3f}} (k = 20); the top-10 % income share from
  {{top10_income_share_%_original|mdav|3}} % to {{top10_income_share_%_anonymized|mdav|20}} % at k = 20.
  Averaging compresses variance (mean variance ratio {{mean_variance_ratio|mdav|3|.2f}} at k = 3,
  {{mean_variance_ratio|mdav|20|.2f}} at k = 20), so dispersion and inequality are slightly understated.
* **Non-linear rules break.** The share of zero-tax filers falls from {{zero_tax_share_%_original|mdav|3}} %
  to {{zero_tax_share_%_anonymized|mdav|20}} % at k = 20 because groups straddling the rebate threshold get a
  positive mean tax, and {{tax_rule_violation_%|mdav|3}} % (k = 3) to {{tax_rule_violation_%|mdav|20}} %
  (k = 20) of released records carry a tax that differs from the schedule applied to their released
  income. Linear identities hold exactly (identity violations {{identity_violation_%|mdav|3}} %),
  whereas individual ranking breaks taxable = gross − deductions for
  {{identity_violation_%|individual_ranking|3}} % of records.
* **Regression.** The R² of the income regression *rises* from {{regression_r2_original|mdav|3|.3f}} to
  {{regression_r2_anonymized|mdav|20|.3f}} at k = 20 – an artefact of averaging out noise, not better
  data – while the mean confidence-interval overlap of the coefficients falls from
  {{regression_ci_overlap|mdav|3|.2f}} (k = 3) to {{regression_ci_overlap|mdav|20|.2f}} (k = 20). In the
  k = {{k_detail}} table, coefficients of categorical attributes used in the grouping (city tier,
  education, occupation) survive comparatively well, whereas the age profile is biased, because
  ages are averaged within groups formed mostly on income and categories.
* **Categories.** Mode replacement changes {{categorical_change_%|mdav|3}} % of categorical cells at
  k = 3 and {{categorical_change_%|mdav|20}} % at k = 20. Rare levels are absorbed by the group mode:
  the gender level "Other" ({{gender_other_original}} taxpayers) survives in {{gender_other_k3}} released
  records at k = 3 and {{gender_other_k5}} at k = 5 – a utility loss for minority analysis, but also a
  privacy gain for a very small, easily identifiable group.

## 13. Privacy–utility trade-off

![Trade-off](../results/figures/tradeoff.png)

![Trade-off, attribute disclosure](../results/figures/tradeoff_attribute.png)

* **MDAV dominates random grouping on utility** at equal k-anonymity: at k = 3 MDAV's IL is
  {{il_percent|mdav|3}} % against {{il_percent|random|3}} %.
* **Individual ranking is the opposite corner**: excellent utility (IL {{il_percent|individual_ranking|3}} %)
  but almost no protection against linkage ({{dbrl_all_%|individual_ranking|3}} %).
* Along the MDAV curve, k = 5 → 10 roughly halves linkage risk
  ({{dbrl_all_%|mdav|5}} % → {{dbrl_all_%|mdav|10}} %) for an IL increase of
  {{il_percent|mdav|5}} % → {{il_percent|mdav|10}} %. Attribute disclosure, however, decreases much more
  slowly ({{interval_disclosure_gross_income_%|mdav|5}} % → {{interval_disclosure_gross_income_%|mdav|10}} %),
  so raising k is an inefficient tool against it.
* **The distance space is a privacy–utility knob too** (ablation, Section 10.4): grouping on raw
  rupees lowers raw IL but sharply raises log-scale IL, i.e. it serves the few richest taxpayers
  at the expense of relative accuracy for everyone else; dropping categories from the distance
  makes incomes very precise – and therefore raises interval disclosure and bracket homogeneity –
  while changing many more categorical values. No single setting is best for both utility
  measures *and* for privacy.
* **Recommendation for this dataset:** MDAV with k = 5 or 10 and the default (log + categorical)
  distance space, released together with documentation of the residual risks, and ideally
  combined with an attribute-disclosure control (Section 16).

## 14. Limitations

1. **No absolute anonymity.** k-anonymity bounds identity disclosure only; attribute disclosure
   (similarity, homogeneity) remains substantial, as Section 11 shows.
2. **Background knowledge.** Knowing the other members of a group reveals the remaining member
   exactly; risk decays as f^(k−1) but never vanishes. Insiders (employers, colleagues, family)
   are realistic sources of such knowledge.
3. **Multiple releases.** Publishing the same population at different k, or with different
   attribute sets, enables differencing attacks. One release per population is advised.
4. **Outliers.** The top earners are few and far apart; their groups are heterogeneous and their
   group means still reveal that a very rich person exists in a given region/occupation.
5. **Heuristic, not optimal.** MDAV is a fixed-size heuristic; the last group can be poor and
   optimal partitions are NP-hard to find.
6. **Utility distortions.** Variances, inequality and non-linear relationships (tax schedule,
   zero-tax mass) are distorted; averaged ages and dependents are fractional; rare categories may
   disappear; regression R² is inflated.
7. **Quadratic cost.** O(n²) time makes plain MDAV slow for millions of returns (blocking or
   approximate nearest neighbours would be needed).
8. **Synthetic data.** Results come from a synthetic population with a stylised tax schedule;
   real data will have different tails, correlations and edit rules.
9. **Guarantees are attribute-specific.** Only the released (aggregated) attributes are covered;
   adding any other attribute to the release voids the k-anonymity argument.

## 15. Conclusion

MDAV micro-aggregation turns a highly sensitive income/tax file into a k-anonymous release that
preserves totals and means exactly and keeps most distributional and policy statistics close to
the original. Compared with univariate micro-aggregation it actually protects against record
linkage, and compared with random grouping it retains far more information. Its O(n²d/k) cost was
confirmed empirically. The experiments also show clearly why micro-aggregation must not be
presented as absolute anonymity: grouping similar taxpayers makes their published incomes close
to their true incomes, homogeneous groups reveal tax brackets, and an attacker with background
knowledge can recover incomes exactly. A responsible release therefore pairs micro-aggregation
with measured risk, conservative k, documented limitations and controlled access.

## 16. Future scope

* **Attribute-disclosure-aware micro-aggregation:** enforce l-diversity / t-closeness of the
  tax bracket or a minimum within-group income spread (e.g. t-closeness-first micro-aggregation,
  Soria-Comas et al., 2015).
* **Variable group sizes** (V-MDAV) and optimal univariate projections to reduce information loss.
* **Hybrid methods:** micro-aggregation followed by calibrated noise, or differentially private
  micro-aggregation, to obtain formal guarantees against background knowledge.
* **Scalability:** blocking by region/occupation, KD-trees or approximate nearest neighbours for
  millions of records.
* **Consistency-preserving post-processing:** recompute tax from released income, or constrained
  aggregation that respects the tax schedule.
* **Evaluation on real administrative data** in a secure environment, and user studies with
  tax analysts on which statistics matter most.

## References

See [`docs/ALGORITHM.md`](ALGORITHM.md#references). Additionally: Defay, D. & Nanopoulos, P.
(1992). Panels of enterprises and confidentiality: the small aggregates method. *Proc. Statistics
Canada Symposium*; Soria-Comas, J., Domingo-Ferrer, J., Sánchez, D. & Martínez, S. (2015).
t-closeness through microaggregation. *IEEE TKDE* 27(11).
