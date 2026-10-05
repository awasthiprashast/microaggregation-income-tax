"""
Release builder and release guard.

``build_release`` is the ONLY function that should produce data leaving the
custodian's environment (CSV export, dashboard download). It

1. keeps only the micro-aggregated attributes (direct identifiers, the derived
   tax bracket and any other column are dropped by construction - whitelist,
   not blacklist);
2. rounds numeric values (2 decimals);
3. shuffles the rows with a seeded permutation and assigns fresh sequential
   ``release_id`` values, so neither row order nor ids can be joined back to
   the source file;
4. runs the release guard and raises ``ReleaseValidationError`` if any blocking
   check fails.

Blocking checks
---------------
* column whitelist       - only released attributes + release_id;
* completeness           - no missing values, same number of records;
* k-anonymity            - every combination of released values occurs >= k
                           times (fails for individual ranking, by design);
* id hygiene             - release_id is a fresh 1..n sequence.

Informational check
-------------------
* exact-value coincidences - how many records have a released sensitive value
  exactly equal to their own original value (possible when a group mean happens
  to equal one member's value). Not blocking, because the value is shared by
  >= k released records, but reported for transparency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .anonymizer import AnonymizationResult
from .config import SENSITIVE_NUMERIC, METHOD_LABELS


class ReleaseValidationError(RuntimeError):
    """Raised when an anonymised dataset is not safe to export."""

    def __init__(self, checks):
        self.checks = checks
        failed = [c for c in checks if c.blocking and not c.passed]
        msg = "; ".join(f"{c.name}: {c.detail}" for c in failed)
        super().__init__(f"Release blocked - {msg}")


@dataclass
class ReleaseCheck:
    name: str
    passed: bool
    detail: str
    blocking: bool = True


def _prepare(result: AnonymizationResult, decimals: int) -> pd.DataFrame:
    cfg = result.config
    rel = result.anonymized[list(cfg.released_attributes)].copy()
    for c in cfg.numeric_attributes:
        rel[c] = rel[c].astype(float).round(decimals)
    order = np.random.default_rng(cfg.seed + 1).permutation(len(rel))
    rel = rel.iloc[order].reset_index(drop=True)
    rel.insert(0, "release_id", np.arange(1, len(rel) + 1))
    return rel


def validate_release(release: pd.DataFrame, result: AnonymizationResult) -> list[ReleaseCheck]:
    cfg = result.config
    checks: list[ReleaseCheck] = []

    allowed = {"release_id", *cfg.released_attributes}
    extra = sorted(set(release.columns) - allowed)
    checks.append(ReleaseCheck(
        "column_whitelist", not extra,
        "only micro-aggregated attributes present" if not extra
        else f"unexpected column(s) would be exported: {extra}"))

    n_missing = int(release.isna().sum().sum())
    same_n = len(release) == len(result.original)
    checks.append(ReleaseCheck(
        "completeness", n_missing == 0 and same_n,
        f"{n_missing} missing cells; {len(release)} of {len(result.original)} records"))

    qi = [c for c in release.columns if c != "release_id"]
    min_class = int(release.groupby(qi, dropna=False).size().min()) if qi else 0
    checks.append(ReleaseCheck(
        "k_anonymity", min_class >= cfg.k,
        f"smallest equivalence class = {min_class} (required >= {cfg.k}); "
        + ("OK" if min_class >= cfg.k else
           f"{METHOD_LABELS[cfg.method]} does not give k-anonymity on the joint attributes")))

    ids_ok = np.array_equal(release["release_id"].to_numpy(), np.arange(1, len(release) + 1))
    checks.append(ReleaseCheck("id_hygiene", ids_ok,
                               "fresh sequential release_id; source identifiers removed"))

    # Informational: released value == own original value.
    coincidences = 0
    sens = [c for c in cfg.numeric_attributes if c in SENSITIVE_NUMERIC]
    for c in sens:
        orig = result.original[c].to_numpy(dtype=float)
        same = np.isclose(result.anonymized[c].round(2).to_numpy(dtype=float), orig)
        coincidences += int((same & (orig != 0)).sum())
    checks.append(ReleaseCheck(
        "exact_value_coincidences", True,
        f"{coincidences} non-zero sensitive cell(s) equal their own original value "
        "(each such value is shared by >= k released records)", blocking=False))
    return checks


def build_release(result: AnonymizationResult, decimals: int = 2,
                  raise_on_fail: bool = True) -> tuple[pd.DataFrame, list[ReleaseCheck]]:
    """Return (release dataframe, checks). Raises ReleaseValidationError if unsafe."""
    rel = _prepare(result, decimals)
    checks = validate_release(rel, result)
    if raise_on_fail and any(c.blocking and not c.passed for c in checks):
        raise ReleaseValidationError(checks)
    return rel, checks


def release_metadata(result: AnonymizationResult) -> dict:
    """Machine-readable release notes to ship alongside the CSV."""
    cfg = result.config
    return {
        "title": "Micro-aggregated income and tax microdata (synthetic)",
        "method": METHOD_LABELS[cfg.method],
        "k": cfg.k,
        "records": int(len(result.original)),
        "released_attributes": list(cfg.released_attributes),
        "numeric_aggregation": "group arithmetic mean",
        "categorical_aggregation": "group mode (ties: alphabetical)",
        "row_order": "randomly permuted; release_id carries no information",
        "notes": [
            "Each record is a group centroid, not a real taxpayer.",
            "Means and totals of numeric attributes are preserved; variances are reduced.",
            "Fractional ages / dependents are averages, not errors.",
            "tax_paid is the group mean of tax paid and generally differs from the "
            "tax computed on the released taxable income (the schedule is non-linear).",
            "Micro-aggregation reduces but does not eliminate disclosure risk.",
        ],
    }
