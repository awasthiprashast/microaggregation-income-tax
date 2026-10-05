"""
Stylised progressive income-tax schedule.

The schedule is *loosely modelled* on Indian personal income-tax slabs so that
the synthetic data has a realistic shape (a large zero-tax mass created by a
rebate, steeply rising marginal rates, surcharge for very high incomes, a flat
rate on capital gains and a health & education cess). It is an illustration
for the privacy experiments only - it is NOT an implementation of any real tax
law and must not be used for tax computation.

All functions are vectorised over numpy arrays.
"""

from __future__ import annotations

import numpy as np

# (upper bound of slab, marginal rate). The last slab is open-ended.
TAX_SLABS: tuple[tuple[float, float], ...] = (
    (300_000, 0.00),
    (600_000, 0.05),
    (900_000, 0.10),
    (1_200_000, 0.15),
    (1_500_000, 0.20),
    (np.inf, 0.30),
)
REBATE_LIMIT: float = 700_000          # slab tax is nil up to this taxable income
CAPITAL_GAINS_EXEMPTION: float = 125_000
CAPITAL_GAINS_RATE: float = 0.125
SURCHARGE: tuple[tuple[float, float], ...] = ((5_000_000, 0.10), (10_000_000, 0.15))
CESS_RATE: float = 0.04

BRACKET_LABELS: tuple[str, ...] = tuple(f"{int(rate * 100)}%" for _, rate in TAX_SLABS)


def slab_tax(taxable_income: np.ndarray) -> np.ndarray:
    """Progressive slab tax: sum over slabs of rate * (income falling in that slab)."""
    income = np.maximum(np.asarray(taxable_income, dtype=float), 0.0)
    tax = np.zeros_like(income)
    lower = 0.0
    for upper, rate in TAX_SLABS:
        in_slab = np.clip(income - lower, 0.0, upper - lower)
        tax += rate * in_slab
        lower = upper
    return tax


def compute_tax(taxable_income: np.ndarray, capital_gains: np.ndarray) -> np.ndarray:
    """
    Total tax liability under the stylised schedule.

    1. slab tax on taxable (non-capital) income;
    2. rebate: nil slab tax up to REBATE_LIMIT, with marginal relief just above it
       (tax can never exceed the income in excess of the limit);
    3. flat tax on capital gains above an exemption;
    4. surcharge on the combined tax for very high total income;
    5. cess on the total.
    """
    taxable = np.maximum(np.asarray(taxable_income, dtype=float), 0.0)
    gains = np.maximum(np.asarray(capital_gains, dtype=float), 0.0)

    base = slab_tax(taxable)
    base = np.where(taxable <= REBATE_LIMIT, 0.0, np.minimum(base, taxable - REBATE_LIMIT))

    gains_tax = CAPITAL_GAINS_RATE * np.maximum(gains - CAPITAL_GAINS_EXEMPTION, 0.0)
    total = base + gains_tax

    total_income = taxable + gains
    surcharge = np.zeros_like(total)
    for threshold, rate in SURCHARGE:
        surcharge = np.where(total_income > threshold, rate, surcharge)
    return total * (1.0 + surcharge) * (1.0 + CESS_RATE)


def tax_bracket(taxable_income: np.ndarray) -> np.ndarray:
    """Marginal slab label (e.g. '15%') for each taxable income."""
    income = np.asarray(taxable_income, dtype=float)
    uppers = np.array([upper for upper, _ in TAX_SLABS])
    idx = np.searchsorted(uppers, income, side="left")
    idx = np.clip(idx, 0, len(BRACKET_LABELS) - 1)
    return np.asarray(BRACKET_LABELS, dtype=object)[idx]
