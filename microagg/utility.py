"""
Utility (information-loss / analytical-validity) metrics.

All metrics compare the cleaned original X with its anonymised version X'
(row-aligned). They return aggregates only - never individual original values.

Information loss (the classic micro-aggregation measure)
    Let z be the attributes standardised with the ORIGINAL means/std. Then
        SST = sum_i ||z_i - zbar||^2      (total sum of squares)
        SSE = sum_i ||z_i - z'_i||^2      (squared error introduced)
        IL  = 100 * SSE / SST  (%)
    For a joint partition with mean aggregation SSE is exactly the
    within-group sum of squares and SST = SSE + SSA (between-group), so
    0 <= IL <= 100. IL = 0 means nothing changed; IL = 100 means every record
    was replaced by the global mean.

Other measures
    * mean preservation, variance ratio Var(X')/Var(X) (always <= 1 for a
      joint partition: averaging removes within-group variance), KS distance;
    * correlation distortion - mean |r_ij - r'_ij| over attribute pairs;
    * categorical change rate - share of cells altered by mode replacement;
    * income-distribution statistics used by tax analysts (Gini, top-10%
      share, zero-tax share, effective tax rate by income quintile);
    * regression validity - confidence-interval overlap of OLS coefficients
      (Karr et al., 2006);
    * tax-rule consistency - does the released tax equal the tax computed on
      the released income? (non-linear rules are not preserved by averaging).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import tax
from .anonymizer import AnonymizationResult
from .features import standardize


# --------------------------------------------------------------------------- #
# Core information loss
# --------------------------------------------------------------------------- #
def information_loss(result: AnonymizationResult) -> dict[str, float]:
    num = list(result.config.numeric_attributes)
    if not num:
        return {"il_percent": float("nan"), "sse": float("nan"), "sst": float("nan")}
    z = standardize(result.original, num)
    z_anon = standardize(result.anonymized, num, reference=result.original)
    sse = float(((z - z_anon) ** 2).sum())
    sst = float(((z - z.mean(axis=0)) ** 2).sum())
    return {"il_percent": 100.0 * sse / sst if sst > 0 else 0.0, "sse": sse, "sst": sst}


def information_loss_log(result: AnonymizationResult) -> float:
    """
    IL computed after log1p() of monetary attributes. On heavy-tailed income
    data the raw-scale IL is dominated by the few largest incomes; the log-scale
    IL instead reflects *relative* accuracy for the typical taxpayer.
    """
    num = list(result.config.numeric_attributes)
    if not num:
        return float("nan")
    o, a = result.original[num].copy(), result.anonymized[num].copy()
    for c in num:
        if c in result.config.monetary_attributes:
            o[c], a[c] = np.log1p(o[c].clip(lower=0)), np.log1p(a[c].clip(lower=0))
    z = standardize(o, num)
    za = standardize(a, num, reference=o)
    sst = float((z ** 2).sum())
    return 100.0 * float(((z - za) ** 2).sum()) / sst if sst > 0 else 0.0


def categorical_change_rate(result: AnonymizationResult) -> dict[str, float]:
    return {
        c: float((result.original[c].astype(str) != result.anonymized[c].astype(str)).mean())
        for c in result.config.categorical_attributes
    }


def attribute_table(result: AnonymizationResult) -> pd.DataFrame:
    """Per-attribute statistical comparison (aggregates only)."""
    rows = []
    for c in result.config.numeric_attributes:
        o = result.original[c].to_numpy(dtype=float)
        a = result.anonymized[c].to_numpy(dtype=float)
        var_o = o.var()
        nz = o != 0
        rows.append({
            "attribute": c,
            "mean_original": o.mean(),
            "mean_anonymized": a.mean(),
            "mean_rel_diff_%": 100 * (a.mean() - o.mean()) / o.mean() if o.mean() else 0.0,
            "variance_ratio": a.var() / var_o if var_o > 0 else 1.0,
            "ks_statistic": float(stats.ks_2samp(o, a, method="asymp").statistic),
            "median_abs_rel_error_%": float(100 * np.median(np.abs(a[nz] - o[nz]) / np.abs(o[nz])))
            if nz.any() else 0.0,
        })
    return pd.DataFrame(rows)


def correlation_distortion(result: AnonymizationResult) -> float:
    num = list(result.config.numeric_attributes)
    if len(num) < 2:
        return 0.0
    r_o = result.original[num].corr().to_numpy()
    r_a = result.anonymized[num].corr().fillna(0).to_numpy()
    iu = np.triu_indices(len(num), 1)
    return float(np.mean(np.abs(r_o[iu] - r_a[iu])))


# --------------------------------------------------------------------------- #
# Income / tax analytics
# --------------------------------------------------------------------------- #
def gini(x: np.ndarray) -> float:
    """Gini coefficient via the sorted-rank formula."""
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    ranks = np.arange(1, n + 1)
    return float((2 * (ranks * x).sum()) / (n * x.sum()) - (n + 1) / n)


def income_statistics(df: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if "gross_income" in df:
        g = df["gross_income"].to_numpy(dtype=float)
        out["gini_gross_income"] = gini(g)
        top = np.sort(g)[-max(1, len(g) // 10):]
        out["top10_income_share_%"] = 100 * top.sum() / g.sum()
    if "tax_paid" in df:
        t = df["tax_paid"].to_numpy(dtype=float)
        out["total_tax_revenue"] = float(t.sum())
        out["zero_tax_share_%"] = float(100 * np.mean(t < 1))
    return out


def effective_tax_by_quintile(df: pd.DataFrame) -> pd.Series | None:
    """Aggregate tax / aggregate gross income within each gross-income quintile."""
    if not {"gross_income", "tax_paid"} <= set(df.columns):
        return None
    q = pd.qcut(df["gross_income"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
    g = df.groupby(q, observed=True)
    return 100 * g["tax_paid"].sum() / g["gross_income"].sum()


def tax_consistency(df: pd.DataFrame) -> dict[str, float]:
    """
    Share of released records violating accounting/tax rules.
    * linear identity taxable = gross - deductions  (preserved by mean aggregation
      unless the max(.,0) floor binds inside a group);
    * tax rule tax_paid = T(taxable, gains)        (NOT preserved: T is non-linear).
    """
    out: dict[str, float] = {}
    cols = set(df.columns)
    if {"gross_income", "deductions", "taxable_income"} <= cols:
        gap = (df["gross_income"] - df["deductions"]).clip(lower=0) - df["taxable_income"]
        out["identity_violation_%"] = float(100 * np.mean(np.abs(gap) > 1.0))
    if {"taxable_income", "capital_gains", "tax_paid"} <= cols:
        recomputed = tax.compute_tax(df["taxable_income"].to_numpy(), df["capital_gains"].to_numpy())
        released = df["tax_paid"].to_numpy(dtype=float)
        bad = np.abs(recomputed - released) > np.maximum(1000.0, 0.05 * released)
        out["tax_rule_violation_%"] = float(100 * bad.mean())
    return out


# --------------------------------------------------------------------------- #
# Regression validity
# --------------------------------------------------------------------------- #
def _design(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Design matrix; dummy levels (and the omitted base level) come from ``reference``
    so that coefficients mean the same thing for original and anonymised data."""
    X = pd.DataFrame(index=df.index)
    X["intercept"] = 1.0
    if "age" in df:
        X["age"] = df["age"]
        X["age_sq/100"] = df["age"] ** 2 / 100
    if "num_dependents" in df:
        X["num_dependents"] = df["num_dependents"]
    for c in ("education", "occupation", "city_tier"):
        if c in df:
            levels = sorted(reference[c].astype(str).unique())
            vals = df[c].astype(str)
            for lvl in levels[1:]:
                X[f"{c}={lvl}"] = (vals == lvl).astype(float)
    return X


def _ols(X: np.ndarray, y: np.ndarray):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, float(r2)


def regression_validity(result: AnonymizationResult) -> dict | None:
    """
    OLS of log(1 + gross income) on demographics, fitted on original and on
    anonymised data. Utility = average confidence-interval overlap J in [0, 1]
    (1 = identical inference). Coefficients for dummy levels that vanish in the
    anonymised data are dropped from the comparison.
    """
    if "gross_income" not in result.config.numeric_attributes:
        return None
    Xo = _design(result.original, result.original)
    Xa = _design(result.anonymized, result.original)
    cols = [c for c in Xo.columns if c in Xa.columns
            and (c == "intercept" or (Xo[c].std() > 0 and Xa[c].std() > 0))]
    if len(cols) < 2:
        return None
    yo = np.log1p(result.original["gross_income"].to_numpy(dtype=float))
    ya = np.log1p(result.anonymized["gross_income"].to_numpy(dtype=float))
    _, _, r2o = _ols(Xo.to_numpy(dtype=float), yo)        # full original model
    bo, so, _ = _ols(Xo[cols].to_numpy(dtype=float), yo)  # restricted to common terms
    ba, sa, r2a = _ols(Xa[cols].to_numpy(dtype=float), ya)

    z = 1.96
    lo_o, hi_o, lo_a, hi_a = bo - z * so, bo + z * so, ba - z * sa, ba + z * sa
    overlap = np.clip(np.minimum(hi_o, hi_a) - np.maximum(lo_o, lo_a), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        J = 0.5 * (overlap / (hi_o - lo_o) + overlap / (hi_a - lo_a))
    J = np.nan_to_num(J)
    table = pd.DataFrame({"coef_original": bo, "coef_anonymized": ba, "ci_overlap": J}, index=cols)
    return {
        "r2_original": r2o,
        "r2_anonymized": r2a,
        "mean_ci_overlap": float(J[1:].mean()) if len(J) > 1 else float(J.mean()),
        "n_coefficients": len(cols),
        "table": table,
    }


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def compute_utility(result: AnonymizationResult) -> dict[str, float]:
    """Flat dict of scalar utility metrics for one anonymisation run."""
    out: dict[str, float] = {}
    out.update(information_loss(result))
    out["il_log_percent"] = information_loss_log(result)
    cat =categorical_change_rate(result)
    out["categorical_change_%"] = 100 * float(np.mean(list(cat.values()))) if cat else 0.0
    out["correlation_distortion"] = correlation_distortion(result)
    attr = attribute_table(result)
    if not attr.empty:
        out["mean_variance_ratio"] = float(attr["variance_ratio"].mean())
        out["mean_ks"] = float(attr["ks_statistic"].mean())
        out["max_abs_mean_diff_%"] = float(attr["mean_rel_diff_%"].abs().max())
        gi = attr.set_index("attribute")
        if "gross_income" in gi.index:
            out["gross_income_median_abs_rel_error_%"] = float(gi.loc["gross_income", "median_abs_rel_error_%"])
    so, sa = income_statistics(result.original), income_statistics(result.anonymized)
    for key in so:
        out[f"{key}_original"] = so[key]
        out[f"{key}_anonymized"] = sa[key]
    out.update(tax_consistency(result.anonymized))
    reg = regression_validity(result)
    if reg:
        out["regression_r2_original"] = reg["r2_original"]
        out["regression_r2_anonymized"] = reg["r2_anonymized"]
        out["regression_ci_overlap"] = reg["mean_ci_overlap"]
    return out
