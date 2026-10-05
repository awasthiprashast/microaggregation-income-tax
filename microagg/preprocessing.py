"""
Input validation, edit rules and missing-value handling.

Pipeline (``validate_and_clean``)
---------------------------------
1. Schema check            - required columns must exist (DataValidationError).
2. Normalisation           - trim/case-fold categorical labels, empty -> missing.
3. De-duplication          - exact duplicate rows and repeated direct identifiers.
4. Type coercion           - non-numeric strings in numeric columns -> missing.
5. Domain checks           - values outside VALID_RANGES -> missing.
6. Logical edit rules      - e.g. deductions may not exceed gross income.
7. Deductive repair        - accounting identity taxable = gross - deductions
                             recovers a value when exactly one of the three is
                             missing (exact, not a guess).
8. Row filter              - rows missing > MAX_MISSING_SHARE_PER_ROW of the
                             analysis attributes are dropped.
9. Imputation              - numeric: median within occupation (fallback: global
                             median); categorical: mode. Tax paid is recomputed
                             from the schedule when it is missing.
10. Derived attributes     - tax bracket is recomputed from the cleaned income.

Every step is counted in a ``ValidationReport`` so the user can see exactly
what was changed. Imputation happens *before* anonymisation: imputed cells are
aggregated like any other cell and are never flagged in the release (a
"was imputed" flag would itself be a disclosure about the respondent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import tax
from .config import (
    CATEGORY_LEVELS,
    DIRECT_IDENTIFIERS,
    INTEGER_ATTRIBUTES,
    MAX_MISSING_SHARE_PER_ROW,
    VALID_RANGES,
)


class DataValidationError(ValueError):
    """Raised when the input cannot be repaired into a usable dataset."""


@dataclass
class ValidationReport:
    n_input_rows: int = 0
    n_output_rows: int = 0
    duplicates_removed: int = 0
    duplicate_ids_removed: int = 0
    rows_dropped_too_sparse: int = 0
    missing_before: dict[str, int] = field(default_factory=dict)
    non_numeric_coerced: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    edit_rule_violations: dict[str, int] = field(default_factory=dict)
    repaired_by_identity: dict[str, int] = field(default_factory=dict)
    imputed: dict[str, int] = field(default_factory=dict)
    unknown_categories: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        """Per-column summary table (counts only - no record values)."""
        cols = sorted(
            set(self.missing_before) | set(self.non_numeric_coerced) | set(self.out_of_range)
            | set(self.edit_rule_violations) | set(self.repaired_by_identity) | set(self.imputed)
        )
        rows = []
        for c in cols:
            rows.append({
                "column": c,
                "missing_in_input": self.missing_before.get(c, 0),
                "non_numeric_coerced": self.non_numeric_coerced.get(c, 0),
                "out_of_range": self.out_of_range.get(c, 0),
                "edit_rule_violations": self.edit_rule_violations.get(c, 0),
                "repaired_by_identity": self.repaired_by_identity.get(c, 0),
                "imputed": self.imputed.get(c, 0),
            })
        return pd.DataFrame(rows)

    def summary(self) -> dict[str, int]:
        return {
            "input_rows": self.n_input_rows,
            "duplicate_rows_removed": self.duplicates_removed,
            "duplicate_ids_removed": self.duplicate_ids_removed,
            "rows_dropped_too_sparse": self.rows_dropped_too_sparse,
            "output_rows": self.n_output_rows,
            "cells_imputed": int(sum(self.imputed.values())),
            "cells_repaired_by_identity": int(sum(self.repaired_by_identity.values())),
        }


def _normalise_categorical(s: pd.Series, levels: tuple[str, ...] | None) -> pd.Series:
    """Trim whitespace, map case-insensitive matches onto the canonical level."""
    out = s.astype("object").where(s.notna(), None)
    out = out.map(lambda v: v.strip() if isinstance(v, str) else v)
    out = out.map(lambda v: None if (isinstance(v, str) and v == "") else v)
    if levels:
        canon = {lvl.lower(): lvl for lvl in levels}
        out = out.map(lambda v: canon.get(v.lower(), v) if isinstance(v, str) else v)
    return out.astype("object")


def validate_and_clean(
    df: pd.DataFrame,
    numeric_attributes: tuple[str, ...] | list[str],
    categorical_attributes: tuple[str, ...] | list[str],
    direct_identifiers: tuple[str, ...] = DIRECT_IDENTIFIERS,
) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Validate and clean ``df``. Returns (clean dataframe, report).

    Only the listed attributes (plus direct identifiers, if present) are
    validated and kept; any other column is dropped so that an un-validated
    column can never leak into a release.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")
    report = ValidationReport(n_input_rows=len(df))
    numeric_attributes = list(numeric_attributes)
    categorical_attributes = list(categorical_attributes)
    analysis_cols = numeric_attributes + categorical_attributes

    missing_cols = [c for c in analysis_cols if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Required column(s) missing from input: {missing_cols}")
    if len(df) == 0:
        raise DataValidationError("Input dataset is empty.")

    id_cols = [c for c in direct_identifiers if c in df.columns]
    extra_known = [c for c in ("tax_bracket",) if c in df.columns and c not in analysis_cols]
    data = df[id_cols + analysis_cols + extra_known].copy()

    # 2. normalise categoricals ------------------------------------------- #
    for c in categorical_attributes:
        data[c] = _normalise_categorical(data[c], CATEGORY_LEVELS.get(c))
        if c in CATEGORY_LEVELS:
            unknown = sorted(set(data[c].dropna()) - set(CATEGORY_LEVELS[c]))
            if unknown:
                report.unknown_categories[c] = unknown[:20]
                report.warnings.append(
                    f"Column '{c}' contains {len(unknown)} unexpected label(s); kept as-is."
                )

    # 3. duplicates ---------------------------------------------------------- #
    before = len(data)
    data = data.drop_duplicates()
    report.duplicates_removed = before - len(data)
    for c in id_cols:
        dup = data[c].notna() & data[c].duplicated(keep="first")
        report.duplicate_ids_removed += int(dup.sum())
        data = data[~dup]
    data = data.reset_index(drop=True)

    report.missing_before = {c: int(data[c].isna().sum()) for c in analysis_cols}

    # 4. numeric coercion ------------------------------------------------- #
    for c in numeric_attributes:
        raw = data[c]
        coerced = pd.to_numeric(raw, errors="coerce")
        report.non_numeric_coerced[c] = int((raw.notna() & coerced.isna()).sum())
        data[c] = coerced.astype(float)
        inf = np.isinf(data[c])
        if inf.any():
            report.non_numeric_coerced[c] += int(inf.sum())
            data.loc[inf, c] = np.nan

    # 5. domain checks -------------------------------------------------------- #
    for c in numeric_attributes:
        lo, hi = VALID_RANGES.get(c, (None, None))
        bad = pd.Series(False, index=data.index)
        if lo is not None:
            bad |= data[c] < lo
        if hi is not None:
            bad |= data[c] > hi
        report.out_of_range[c] = int(bad.sum())
        data.loc[bad, c] = np.nan

    has = set(numeric_attributes).__contains__

    # 6. edit rules ------------------------------------------------------------ #
    if has("deductions") and has("gross_income"):
        bad = data["deductions"] > data["gross_income"]
        report.edit_rule_violations["deductions"] = int(bad.sum())
        data.loc[bad, "deductions"] = np.nan
    if has("taxable_income") and has("gross_income"):
        bad = data["taxable_income"] > data["gross_income"]
        report.edit_rule_violations["taxable_income"] = int(bad.sum())
        data.loc[bad, "taxable_income"] = np.nan

    # 7. deductive repair: taxable = max(gross - deductions, 0) ----------------- #
    if has("gross_income") and has("deductions") and has("taxable_income"):
        g, d, t = data["gross_income"], data["deductions"], data["taxable_income"]
        fix_t = t.isna() & g.notna() & d.notna()
        data.loc[fix_t, "taxable_income"] = np.maximum(g[fix_t] - d[fix_t], 0.0)
        # deductions / gross can only be recovered exactly when taxable > 0
        fix_d = d.isna() & g.notna() & t.notna() & (t > 0)
        data.loc[fix_d, "deductions"] = g[fix_d] - t[fix_d]
        fix_g = g.isna() & d.notna() & t.notna() & (t > 0)
        data.loc[fix_g, "gross_income"] = t[fix_g] + d[fix_g]
        report.repaired_by_identity = {
            "taxable_income": int(fix_t.sum()),
            "deductions": int(fix_d.sum()),
            "gross_income": int(fix_g.sum()),
        }

    # 8. drop rows that are mostly empty ---------------------------------------- #
    share_missing = data[analysis_cols].isna().mean(axis=1)
    too_sparse = share_missing > MAX_MISSING_SHARE_PER_ROW
    report.rows_dropped_too_sparse = int(too_sparse.sum())
    data = data[~too_sparse].reset_index(drop=True)

    # 9. imputation ------------------------------------------------------------- #
    group_key = "occupation" if "occupation" in categorical_attributes else None
    # categorical first so that the occupation used for grouping is complete
    for c in categorical_attributes:
        n_missing = int(data[c].isna().sum())
        if n_missing:
            mode = data[c].mode(dropna=True)
            if mode.empty:
                raise DataValidationError(f"Column '{c}' has no observed values to impute from.")
            data[c] = data[c].fillna(sorted(mode)[0])
        report.imputed[c] = n_missing

    imputed_mask: dict[str, pd.Series] = {}
    for c in numeric_attributes:
        if c == "tax_paid":
            continue  # handled below by recomputation when possible
        miss = data[c].isna()
        imputed_mask[c] = miss
        n_missing = int(miss.sum())
        if n_missing:
            global_median = data[c].median()
            if np.isnan(global_median):
                raise DataValidationError(f"Column '{c}' has no valid values to impute from.")
            if group_key:
                fill = data.groupby(group_key)[c].transform("median").fillna(global_median)
            else:
                fill = pd.Series(global_median, index=data.index)
            data.loc[miss, c] = fill[miss]
        report.imputed[c] = n_missing

    # Median imputation can break taxable = max(gross - deductions, 0). Restore it
    # on imputed rows only, always keeping the *observed* values fixed.
    if has("gross_income") and has("deductions") and has("taxable_income"):
        ig, idd, it = (imputed_mask["gross_income"], imputed_mask["deductions"],
                       imputed_mask["taxable_income"])
        g_from_t = ig & ~it                       # observed taxable -> rebuild gross
        data.loc[g_from_t, "gross_income"] = (
            data.loc[g_from_t, "taxable_income"] + data.loc[g_from_t, "deductions"])
        data["deductions"] = np.minimum(data["deductions"], data["gross_income"])
        t_rows = it | idd
        data.loc[t_rows, "taxable_income"] = np.maximum(
            data.loc[t_rows, "gross_income"] - data.loc[t_rows, "deductions"], 0.0)

    if has("tax_paid"):
        miss = data["tax_paid"].isna()
        if miss.any():
            if has("taxable_income") and has("capital_gains"):
                data.loc[miss, "tax_paid"] = np.round(tax.compute_tax(
                    data.loc[miss, "taxable_income"], data.loc[miss, "capital_gains"]))
            else:
                data.loc[miss, "tax_paid"] = data["tax_paid"].median()
        report.imputed["tax_paid"] = int(miss.sum())

    for c in INTEGER_ATTRIBUTES:
        if c in numeric_attributes:
            data[c] = np.round(data[c])

    # 10. derived sensitive attribute ------------------------------------------- #
    if has("taxable_income"):
        data["tax_bracket"] = tax.tax_bracket(data["taxable_income"].to_numpy())
    elif "tax_bracket" in data.columns:
        data = data.drop(columns="tax_bracket")

    if data[analysis_cols].isna().any().any():  # defensive: should be impossible
        raise DataValidationError("Missing values remain after imputation.")

    report.n_output_rows = len(data)
    if report.n_output_rows < 0.5 * report.n_input_rows:
        report.warnings.append("More than half of the input rows were removed during cleaning.")
    return data, report
