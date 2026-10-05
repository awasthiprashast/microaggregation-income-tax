"""
Helpers that make sure ORIGINAL confidential values never reach a screen or
file in exact form unless the custodian explicitly asks for it.

Rules applied
-------------
* Original monetary values are shown as coarse bands (e.g. "Rs 5L-7.5L").
* Original ages are shown as 10-year bands.
* Summaries of original data report count/mean/std/quartiles rounded to three
  significant figures and NEVER min/max (min and max are single persons' values).
* Histograms of original data are top-coded at the 99th percentile so an
  outlier cannot be read off the axis.
* Record-level comparisons never show per-record errors: released value +
  error (or ratio) would let anyone back out the exact original value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .anonymizer import AnonymizationResult
from .config import MONETARY_ATTRIBUTES

MONEY_EDGES = (0, 2.5e5, 5e5, 7.5e5, 1e6, 1.5e6, 2.5e6, 5e6, 1e7, np.inf)


def format_inr(x: float) -> str:
    """Indian-style compact currency: 1.25L = 1.25 lakh, 1.2Cr = 1.2 crore."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    if x >= 1e7:
        return f"₹{x / 1e7:.2f}Cr"
    if x >= 1e5:
        return f"₹{x / 1e5:.2f}L"
    return f"₹{x:,.0f}"


def _edge_label(v: float) -> str:
    if v == 0:
        return "0"
    if v >= 1e7:
        return f"{v / 1e7:g}Cr"
    return f"{v / 1e5:g}L"


def band_money(values: pd.Series) -> pd.Series:
    labels = []
    for lo, hi in zip(MONEY_EDGES[:-1], MONEY_EDGES[1:]):
        labels.append(f"≥ ₹{_edge_label(lo)}" if np.isinf(hi) else f"₹{_edge_label(lo)}–{_edge_label(hi)}")
    cut = pd.cut(values.astype(float), bins=list(MONEY_EDGES), labels=labels, right=False,
                 include_lowest=True)
    out = cut.astype(object)
    out[values.astype(float) == 0] = "₹0"
    return out


def band_age(values: pd.Series) -> pd.Series:
    lo = (np.floor(values.astype(float) / 10) * 10).astype(int)
    return lo.astype(str) + "–" + (lo + 9).astype(str)


def _sig(x: float, digits: int = 3) -> float:
    if x == 0 or not np.isfinite(x):
        return x
    return float(round(x, -int(math.floor(math.log10(abs(x)))) + (digits - 1)))


def safe_summary(df: pd.DataFrame, columns) -> pd.DataFrame:
    rows = []
    for c in columns:
        v = df[c].astype(float)
        rows.append({
            "attribute": c, "count": int(v.count()), "mean": _sig(v.mean()), "std": _sig(v.std()),
            "P25": _sig(v.quantile(0.25)), "median": _sig(v.median()), "P75": _sig(v.quantile(0.75)),
        })
    return pd.DataFrame(rows)


def top_code(values: pd.Series, q: float = 0.99) -> pd.Series:
    return values.clip(upper=values.quantile(q))


def comparison_table(result: AnonymizationResult, n: int = 10, seed: int = 0,
                     reveal_original: bool = False) -> pd.DataFrame:
    """
    Side-by-side original vs released values for ``n`` random records.

    With reveal_original=False (default) original monetary values are banded and
    ages are bucketed; categorical originals are shown as-is. Rows are labelled
    'Sample 1..n' - no identifier of the source file is ever shown.
    """
    cfg = result.config
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(result.original), size=min(n, len(result.original)), replace=False))
    orig, anon = result.original.iloc[idx], result.anonymized.iloc[idx]
    out = pd.DataFrame(index=[f"Sample {i + 1}" for i in range(len(idx))])
    for c in cfg.released_attributes:
        o = orig[c].reset_index(drop=True)
        a = anon[c].reset_index(drop=True)
        if c in cfg.numeric_attributes:
            if reveal_original:
                o_disp = o.round(2)
            elif c in MONETARY_ATTRIBUTES:
                o_disp = band_money(o)
            elif c == "age":
                o_disp = band_age(o)
            else:
                o_disp = o.round(0)  # small counts such as dependents
            a_disp = a.map(format_inr) if c in MONETARY_ATTRIBUTES else a.round(2)
        else:
            o_disp, a_disp = o, a
        out[f"{c} · original"] = o_disp.to_numpy()
        out[f"{c} · released"] = a_disp.to_numpy()
    if result.is_joint:
        out["group size"] = result.group_sizes[result.labels[idx]]
    return out
