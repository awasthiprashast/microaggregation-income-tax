# Micro-aggregation and the MDAV algorithm

This document explains every algorithm used in the project: what it does mathematically,
why it works intuitively, what it costs, and what it does **not** protect against.
Code references point to the `microagg/` package.

---

## 1. Setting and notation

| Symbol | Meaning |
|---|---|
| $X = \{x_1,\dots,x_n\}$ | the microdata file, one record per taxpayer |
| $x_i \in \mathbb{R}^d$ | record $i$ in the *distance space* (Section 3) |
| $k$ | minimum group size (privacy parameter) |
| $G_1,\dots,G_g$ | a partition of the records into groups |
| $\bar{x}_j$ | centroid (mean) of group $G_j$; $\bar{x}$ is the global centroid |
| $x'_i$ | the released (masked) version of record $i$ |

Attributes have three roles (`microagg/config.py`):

* **direct identifier** – `taxpayer_id`; never released;
* **quasi-identifiers (QIs)** – age, dependents, gender, marital status, education, occupation,
  region, city tier; harmless alone but linkable to external sources;
* **confidential attributes** – gross income, deductions, taxable income, capital gains, tax paid.
  The derived *tax bracket* is confidential too and is **not released** (it would reveal the
  un-aggregated income).

## 2. Micro-aggregation as an optimisation problem

Micro-aggregation masks data in two steps:

1. **Partition** the records into groups of at least $k$ similar records.
2. **Aggregate**: replace every record by its group centroid, $x'_i = \bar{x}_{j}$ for $i \in G_j$.

The best partition is the one that loses the least information, measured by the within-group
sum of squares

$$
\mathrm{SSE} = \sum_{j=1}^{g} \sum_{x_i \in G_j} \lVert x_i - \bar{x}_j \rVert^2
\qquad\text{subject to } |G_j| \ge k .
$$

Because the total sum of squares decomposes as

$$
\mathrm{SST} = \sum_i \lVert x_i - \bar{x} \rVert^2 = \mathrm{SSE} + \mathrm{SSA},\qquad
\mathrm{SSA} = \sum_j |G_j|\,\lVert \bar{x}_j - \bar{x} \rVert^2 ,
$$

the normalised **information loss** $\mathrm{IL} = 100 \cdot \mathrm{SSE}/\mathrm{SST}$ lies in $[0, 100]$:
0 means nothing changed, 100 means every record became the global mean.

**Why the mean?** For a fixed group, $f(c) = \sum_{i\in G}\lVert x_i - c\rVert^2$ has gradient
$-2\sum_{i}(x_i - c)$, which vanishes at $c = \bar{x}_G$. The mean is therefore the
SSE-optimal representative. For categorical attributes the analogous minimiser of the number of
mismatches is the **mode** (`mdav.aggregate`).

**Why groups of size $k$ to $2k-1$?** Any group with $\ge 2k$ records can be split into two
groups of size $\ge k$ without increasing SSE (the SSE of a set is at least the sum of the SSEs of
its parts, each around its own mean). So an optimal partition only needs sizes in $[k, 2k-1]$
(Domingo-Ferrer & Mateo-Sanz, 2002).

**Hardness.** Finding the optimal multivariate partition is NP-hard (Oganian & Domingo-Ferrer, 2001).
Only the univariate case has a polynomial optimum (Hansen & Mukherjee, 2003). Hence heuristics.

## 3. The distance space (`microagg/features.py`)

Taxpayer records mix rupees, years and categories, so they are mapped to a common space:

$$
\phi(x) = \Big(\underbrace{\tfrac{t_1(x_1)-\mu_1}{\sigma_1},\dots}_{\text{numeric}},\;
\underbrace{w\cdot\mathrm{onehot}(x_c),\dots}_{\text{categorical}}\Big),
\qquad t_j = \begin{cases}\log(1+x) & \text{monetary}\\ x & \text{otherwise}\end{cases}
$$

* **Standardisation** makes a 1-σ difference in age as important as a 1-σ difference in income.
* **log1p for money**: incomes are heavy-tailed. On the raw scale a single crore-earner is
  dozens of σ from the mean and dominates every distance; on the log scale distance reflects
  *relative* differences, which is what matters for most taxpayers.
* **Categorical weight $w$**: two different levels are $\sqrt{2}\,w$ apart, so $w=1/\sqrt{2}$
  makes a category mismatch cost exactly as much as a 1-σ numeric difference.

The transforms are used **only to decide who is grouped with whom**. Released numeric values
are always raw-scale group means, so totals are preserved.

## 4. MDAV – Maximum Distance to Average Vector (`microagg/mdav.py::mdav`)

### 4.1 Pseudo-code

```
MDAV(X, k):
    R ← X
    while |R| ≥ 3k:
        c  ← centroid(R)
        r  ← argmax_{x∈R} ‖x − c‖          # most extreme record
        s  ← argmax_{x∈R} ‖x − r‖          # most extreme w.r.t. r (opposite side)
        G  ← {r} ∪ (k−1 nearest neighbours of r in R);   R ← R \ G
        G' ← {s} ∪ (k−1 nearest neighbours of s in R);   R ← R \ G'
    if |R| ≥ 2k:
        c ← centroid(R);  r ← argmax_{x∈R} ‖x − c‖
        G ← {r} ∪ (k−1 nearest neighbours of r in R);    R ← R \ G
    form a final group with the remaining records       # k ≤ |R| ≤ 2k−1
```

Implementation details:

* `s` is chosen among records not already grouped with `r` (only matters for degenerate data
  where many records coincide).
* k-nearest-neighbour selection uses `np.partition` (linear time) and breaks ties by the lowest
  original row index, so the algorithm is **fully deterministic**: same input ⇒ same partition.
* Squared distances are used throughout (same ordering, no square roots).

### 4.2 Intuition

MDAV works **from the outside in**. Records far from the centre are the hardest to hide: they
have few close neighbours. Handling them first, while the pool of candidates is still large,
lets them be grouped with their genuinely closest neighbours. Picking `r` and then `s` (the
record farthest from `r`) builds two groups at opposite ends of the cloud in each iteration, so
consecutive groups do not compete for the same neighbours. The remaining central records are
the easiest to group well at the end.

### 4.3 Worked example (k = 3, 9 points in 2-D)

Points: $x_0..x_8$ = (1,1), (2,1), (1,2), (8,8), (9,8), (8,9), (5,5), (4,6), (5,4).
(These are the exact values used by the code; you can reproduce them with `mdav(X, 3)`.)

1. $|R|=9 \ge 3k$. Centroid $c=(4.78, 4.89)$. Squared distances to $c$ are largest for
   $x_0=(1,1)$ (29.4) ⇒ $r=x_0$.
2. Farthest from $r$: $x_4=(9,8)$ and $x_5=(8,9)$ tie at 113; the lower index wins ⇒ $s=x_4$.
3. $G_1 = \{x_0, x_1, x_2\}$ (distances 0, 1, 1 from $r$). Centroid (1.33, 1.33).
4. $G_2 = \{x_4, x_3, x_5\}$ (distances 0, 1, 2 from $s$). Centroid (8.33, 8.33).
5. $|R| = 3 < 2k$ ⇒ final group $G_3 = \{x_6, x_7, x_8\}$, centroid (4.67, 5.00).

$\mathrm{SSE}=1.33+1.33+2.67=5.33$, $\mathrm{SST}=152.44$, so $\mathrm{IL}=3.5\%$. Every released
point is shared by exactly 3 records (3-anonymity). With only 8 points (drop $x_8$) the loop
never runs; one group of 3 is formed around (1,1) and the other five records — including the
middle points (5,5), (4,6) — are merged into the (8,8) cluster, a reminder that the last group
can be poor.

### 4.4 Complexity analysis

Let $m$ be the number of unassigned records at the start of an iteration.

| Step | Cost |
|---|---|
| centroid of $R$ | $O(md)$ |
| distances to the centroid, argmax | $O(md)$ |
| distances to $r$, argmax for $s$ | $O(md)$ |
| k-NN of $r$ via linear-time selection | $O(m)$ |
| distances to $s$ + k-NN of $s$ | $O(md)$ |

So one iteration costs $\Theta(md)$ and removes $2k$ records. Summing over the
$\lfloor n/(2k) \rfloor$ iterations:

$$
T(n) = \sum_{i=0}^{n/(2k)} c\,d\,(n-2ki) \;\approx\; c\,d\,\frac{n^2}{4k}
\quad\Longrightarrow\quad T(n) = O\!\left(\frac{n^2 d}{k}\right).
$$

Memory is $O(nd)$ for the data plus $O(n)$ for labels and distance vectors. For fixed $d$ and $k$
this is the familiar $O(n^2)$ bound of the literature. Aggregation afterwards is $O(nd)$.

**Empirical check** (`experiments.scalability`, figure `results/figures/runtime.png`): the log–log
slope of runtime against $n$ is ≈ 2 and against $k$ is ≈ −1, matching the analysis.

Comparison with the other partitioners in this project:

| Method | Time | Guarantees k-anonymity on the joint attributes? |
|---|---|---|
| MDAV | $O(n^2 d/k)$ | Yes |
| Individual ranking (per attribute) | $O(d\,n\log n)$ | **No** |
| Random partition | $O(n)$ | Yes (but huge information loss) |

## 5. Baselines

**Individual ranking (IR)** (`mdav.univariate_groups`). Each attribute is micro-aggregated on its
own: sort, cut the ranking into blocks of $k$, replace by block means. Because only
neighbours *in one attribute* are averaged, IR loses very little information — but different
attributes use different groupings, so the *combination* of released values stays almost unique.
It does not give k-anonymity and it breaks even linear identities (gross − deductions ≠ taxable).

**Random partition** (`mdav.random_partition`). Same group sizes as MDAV, but members are chosen
at random. It provides the same k-anonymity guarantee and shows how much utility MDAV's
similarity-based grouping buys.

## 6. What micro-aggregation guarantees

**Claim (k-anonymity).** If all released attributes are aggregated with the *same* partition,
every released record is identical to at least $k-1$ others.

*Proof.* All records of $G_j$ receive the same centroid (mean for numbers, mode for categories)
and $|G_j|\ge k$. ∎

Consequence for identity disclosure: an attacker who links a target to the released file can at
best pick one of $\ge k$ identical records, so the probability of picking the right one is
$\le 1/k$. This is what the distance-based record-linkage metric measures
(`privacy.distance_based_linkage`): success for record $i$ is $1/m$ if the nearest released vector
is its own (shared by $m$ records) and 0 otherwise.

Utility guarantees: means and totals of every numeric attribute are preserved exactly
(e.g. total tax revenue), and linear relations shared by all members of a group (such as
taxable = gross − deductions) survive. Variances shrink by the within-group share.

## 7. What micro-aggregation does NOT guarantee — attacks

Micro-aggregation does not provide absolute anonymity. The project quantifies five threats
(`microagg/privacy.py`).

1. **Attribute disclosure via similarity.** MDAV puts *similar* taxpayers together, so their
   incomes are similar. Learning a person's group (from their quasi-identifiers) reveals their
   income approximately. Metric: share of records whose released value is within ±p % of the true
   value (`interval_disclosure`), and an attacker who links on QIs only and reads off the income
   (`attribute_inference_attack`), compared with a no-release baseline.

2. **Homogeneity attack.** If all members of a group share the same tax bracket, identifying the
   group reveals the bracket with certainty. Metrics: share of records in homogeneous groups,
   distinct *l*-diversity and *t*-closeness of the bracket (ordered EMD,
   $\mathrm{EMD} = \frac{1}{m-1}\sum_i |\sum_{j\le i}(p_j - q_j)|$).

3. **Dominance (outliers).** If one member's income dominates a group, the published mean is
   roughly that person's income divided by $|G|$. Tax agencies test this with the
   (1, 50 %) dominance rule ($x_{(1)} > 0.5\sum x$) and the p %-rule
   ($\sum x - x_{(1)} - x_{(2)} < \frac{p}{100}x_{(1)}$) (`dominance_rules`).

4. **Background-knowledge (collusion / difference) attack.** The released mean gives the group
   total $S = |G|\,\bar{x}_G$. An attacker who knows the true values of all other members computes
   $x_t = S - \sum_{j\ne t} x_j$ **exactly**. If the attacker knows a random share $f$ of the
   population, the chance that all $|G|-1$ others are known is about $f^{|G|-1}$: it falls
   exponentially in $k$ but is never zero (`collusion_attack`).

5. **Multiple releases / differencing.** Publishing the same records at two values of $k$, or with
   two attribute sets, gives two different partitions whose means can be combined to narrow down
   individual values. Not simulated, but it is the reason the release notes advise a single
   release per population.

Other limitations: k-anonymity says nothing about an attacker with arbitrary auxiliary
information (unlike differential privacy); the guarantee holds only for the attributes that were
aggregated; and imputed or synthetic values still describe real-looking people.

## 8. Measuring utility

| Metric | Definition | Ideal |
|---|---|---|
| IL (raw) | $100\cdot\mathrm{SSE}/\mathrm{SST}$ on z-scored raw attributes | 0 |
| IL (log) | same after log1p of monetary attributes | 0 |
| Variance ratio | $\mathrm{Var}(X')/\mathrm{Var}(X)$ per attribute | 1 |
| KS statistic | max distance between empirical CDFs | 0 |
| Correlation distortion | mean $\lvert r_{ab}-r'_{ab}\rvert$ | 0 |
| Categorical change | share of categorical cells altered by the mode | 0 |
| CI overlap | average overlap of 95 % CIs of OLS coefficients (Karr et al., 2006) | 1 |
| Tax-rule violation | share of records where released tax ≠ schedule(released income) | 0 |
| Gini, top-10 % share, zero-tax share, effective tax rate by quintile | policy statistics | unchanged |

Note on raw vs log IL: on heavy-tailed income data the top 1 % of earners contribute most of
SST, so raw IL mostly measures how well the richest few are grouped. Log IL measures relative
accuracy for the typical taxpayer. The ablation in the report shows that the distance-space
choice trades one against the other.

## References

* Domingo-Ferrer, J. & Mateo-Sanz, J. M. (2002). Practical data-oriented microaggregation for
  statistical disclosure control. *IEEE TKDE* 14(1).
* Domingo-Ferrer, J. & Torra, V. (2005). Ordinal, continuous and heterogeneous k-anonymity
  through microaggregation. *Data Mining and Knowledge Discovery* 11(2).
* Oganian, A. & Domingo-Ferrer, J. (2001). On the complexity of optimal microaggregation for
  statistical disclosure control. *Statistical Journal of the UN ECE* 18.
* Hansen, S. L. & Mukherjee, S. (2003). A polynomial algorithm for optimal univariate
  microaggregation. *IEEE TKDE* 15(4).
* Sweeney, L. (2002). k-anonymity: a model for protecting privacy. *IJUFKS* 10(5).
* Machanavajjhala, A. et al. (2007). l-diversity: privacy beyond k-anonymity. *ACM TKDD* 1(1).
* Li, N., Li, T. & Venkatasubramanian, S. (2007). t-closeness: privacy beyond k-anonymity and
  l-diversity. *ICDE*.
* Karr, A. F. et al. (2006). A framework for evaluating the utility of data altered to protect
  confidentiality. *The American Statistician* 60(3).
* Hundepool, A. et al. (2012). *Statistical Disclosure Control*. Wiley.
* Dwork, C. (2006). Differential privacy. *ICALP*.
