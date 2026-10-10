"""
Render docs/report_template.md into docs/REPORT.md using the files written by
run_experiments.py, so every number in the report comes from an actual run.

Placeholder syntax inside the template
--------------------------------------
{{table:NAME}}                  a Markdown table built from results (see TABLES)
{{metric|method|k}}             value from results/summary.csv, formatted .1f
{{metric|method|k|FMT}}         same with an explicit format spec (e.g. .3f, .0f)
{{scalar}} / {{scalar|FMT}}     value from results/run_info.json

An unknown placeholder raises KeyError, so template and results cannot drift
apart silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METHOD_LABELS

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def md_table(df: pd.DataFrame, floatfmt: str = ".2f", index: bool = False) -> str:
    if index:
        df = df.reset_index()
    cols = [str(c) for c in df.columns]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "–" if np.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _tables(res_dir: Path, summary: pd.DataFrame) -> dict[str, str]:
    s = summary.copy()
    s["Method"] = s["method"].map(METHOD_LABELS)
    t: dict[str, str] = {}

    t["utility"] = md_table(s[[
        "Method", "k", "il_percent", "il_log_percent", "categorical_change_%",
        "correlation_distortion", "mean_variance_ratio", "regression_ci_overlap",
        "tax_rule_violation_%"]].rename(columns={
            "il_percent": "IL raw (%)", "il_log_percent": "IL log (%)",
            "categorical_change_%": "Cat. changed (%)", "correlation_distortion": "Corr. distortion",
            "mean_variance_ratio": "Var. ratio", "regression_ci_overlap": "CI overlap",
            "tax_rule_violation_%": "Tax-rule violations (%)"}), ".2f")

    t["privacy"] = md_table(s[[
        "Method", "k", "k_anonymity_level", "dbrl_all_%", "dbrl_qi_%",
        "interval_disclosure_gross_income_%", "inference_success_%",
        "homogeneous_records_%", "mean_l_diversity", "p_rule_records_%",
        "collusion_f0.5_%"]].rename(columns={
            "k_anonymity_level": "Min. class", "dbrl_all_%": "DBRL all (%)",
            "dbrl_qi_%": "DBRL QI (%)", "interval_disclosure_gross_income_%": "Income ±10% (%)",
            "inference_success_%": "QI inference (%)", "homogeneous_records_%": "Bracket homog. (%)",
            "mean_l_diversity": "Mean l", "p_rule_records_%": "p%-rule (%)",
            "collusion_f0.5_%": "Collusion f=0.5 (%)"}), ".2f")

    stats_cols = ["gini_gross_income", "top10_income_share_%", "total_tax_revenue", "zero_tax_share_%"]
    rows = []
    base = s.iloc[0]
    rows.append({"Dataset": "Original", **{c: base[f"{c}_original"] for c in stats_cols}})
    for _, r in s[s["method"] == "mdav"].iterrows():
        rows.append({"Dataset": f"MDAV k={int(r['k'])}", **{c: r[f"{c}_anonymized"] for c in stats_cols}})
    st = pd.DataFrame(rows)
    st["total_tax_revenue"] = st["total_tax_revenue"] / 1e7
    t["income_stats"] = md_table(st.rename(columns={
        "gini_gross_income": "Gini (gross income)", "top10_income_share_%": "Top-10% share (%)",
        "total_tax_revenue": "Total tax (₹ crore)", "zero_tax_share_%": "Zero-tax filers (%)"}), ".3f")

    for name in ("ablation", "collusion", "scalability", "validation_report",
                 "record_comparison_mdav_k5", "attribute_mdav_k5", "regression_mdav_k5"):
        path = res_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path)
            fmt = ".3f" if name in ("scalability", "regression_mdav_k5") else ".2f"
            t[name] = md_table(df.rename(columns=_pretty), fmt)
    return t


PRETTY_HEADERS = {
    "attribute": "Attribute", "mean_original": "Mean (original)", "mean_anonymized": "Mean (released)",
    "mean_rel_diff_%": "Mean diff. (%)", "variance_ratio": "Variance ratio", "ks_statistic": "KS statistic",
    "median_abs_rel_error_%": "Median abs. rel. error (%)",
    "term": "Term", "coef_original": "Coef. (original)", "coef_anonymized": "Coef. (released)",
    "ci_overlap": "CI overlap",
    "distance_space": "Distance space", "il_percent": "IL raw (%)", "il_log_percent": "IL log (%)",
    "categorical_change_%": "Cat. changed (%)",
    "gross_income_median_abs_rel_error_%": "Income median rel. error (%)",
    "interval_disclosure_gross_income_%": "Income ±10% (%)",
    "homogeneous_records_%": "Bracket homog. (%)", "dbrl_all_%": "DBRL all (%)",
    "known_fraction": "Known share f", "exposed_targets_%": "Exposed targets (%)",
    "exactly_recovered_%": "Recovered exactly (%)", "expected_%": "Expected (%)",
    "sweep": "Sweep", "seconds": "Seconds",
    "column": "Column", "missing_in_input": "Missing", "non_numeric_coerced": "Non-numeric",
    "out_of_range": "Out of range", "edit_rule_violations": "Rule violations",
    "repaired_by_identity": "Repaired (identity)", "imputed": "Imputed",
    "record": "Record", "group size": "Group size",
}


def _pretty(col: str) -> str:
    """Human-readable table header (no '|' characters - they would break Markdown tables)."""
    if col in PRETTY_HEADERS:
        return PRETTY_HEADERS[col]
    if " · " in col:                                   # record-comparison columns
        attr, side = col.split(" · ")
        return f"{attr.replace('_', ' ').capitalize()} ({'orig.' if side == 'original' else 'rel.'})"
    return col


def build_report(template_path: Path, res_dir: Path, out_path: Path) -> Path:
    summary = pd.read_csv(res_dir / "summary.csv")
    info = json.loads((res_dir / "run_info.json").read_text(encoding="utf-8"))
    tables = _tables(res_dir, summary)
    lookup = summary.set_index(["method", "k"])

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key.startswith("table:"):
            return tables[key[len("table:"):]]
        parts = key.split("|")
        if len(parts) >= 3:
            metric, method, k = parts[0], parts[1], int(parts[2])
            fmt = parts[3] if len(parts) > 3 else ".1f"
            return format(float(lookup.loc[(method, k), metric]), fmt)
        value = info[parts[0]]
        return format(value, parts[1]) if len(parts) > 1 else str(value)

    text = PLACEHOLDER.sub(sub, template_path.read_text(encoding="utf-8"))
    out_path.write_text(text, encoding="utf-8")
    return out_path
