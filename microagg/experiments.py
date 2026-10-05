"""
Experiment drivers.

* run_grid        - every (method, k) combination on the same cleaned data;
* collusion_curves- background-knowledge attack success vs. attacker knowledge;
* ablation        - effect of the distance-space design (log / categorical weight);
* scalability     - empirical MDAV runtime vs. n and vs. k (checks O(n^2 / k)).

Nothing here hard-codes a result: every number is computed from the data.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd

from . import privacy, utility
from .anonymizer import AnonymizationResult, anonymize
from .config import AnonymizationConfig, EXPERIMENT_K_VALUES, METHODS
from .data_generator import generate_income_tax_data
from .mdav import mdav
from .preprocessing import validate_and_clean


def evaluate(result: AnonymizationResult) -> dict:
    cfg = result.config
    row = {
        "method": cfg.method,
        "k": cfg.k,
        "n_records": len(result.original),
        "n_groups": result.n_groups if result.n_groups is not None else np.nan,
        "max_group_size": int(result.group_sizes.max()) if result.is_joint else np.nan,
        "runtime_s": result.runtime_seconds,
    }
    row.update(utility.compute_utility(result))
    row.update(privacy.compute_privacy(result))
    return row


def run_grid(clean: pd.DataFrame, k_values=EXPERIMENT_K_VALUES, methods=METHODS,
             base: AnonymizationConfig | None = None, progress=None):
    """Returns (summary DataFrame, {(method, k): AnonymizationResult})."""
    base = base or AnonymizationConfig()
    rows, results = [], {}
    for method in methods:
        for k in k_values:
            res = anonymize(clean, replace(base, method=method, k=int(k)))
            results[(method, int(k))] = res
            rows.append(evaluate(res))
            if progress:
                progress(method, k)
    return pd.DataFrame(rows), results


def collusion_curves(results: dict, method: str = "mdav",
                     fractions=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame:
    frames = []
    for (m, k), res in sorted(results.items()):
        if m != method:
            continue
        df = privacy.collusion_attack(res, fractions=fractions)
        if df is not None:
            df.insert(0, "k", k)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ablation(clean: pd.DataFrame, k: int = 5, base: AnonymizationConfig | None = None) -> pd.DataFrame:
    """MDAV with different distance-space designs."""
    base = base or AnonymizationConfig()
    variants = [
        ("log + categorical (default)", True, base.categorical_weight),
        ("log, numeric only in distance", True, 0.0),
        ("raw scale + categorical", False, base.categorical_weight),
        ("raw scale, numeric only", False, 0.0),
    ]
    rows = []
    for name, log, w in variants:
        res = anonymize(clean, replace(base, k=k, method="mdav",
                                       log_transform_monetary=log, categorical_weight=w))
        u = utility.compute_utility(res)
        inter = privacy.interval_disclosure(res)
        hom = privacy.homogeneity(res) or {}
        rows.append({
            "distance_space": name,
            "il_percent": u["il_percent"],
            "il_log_percent": u["il_log_percent"],
            "categorical_change_%": u["categorical_change_%"],
            "gross_income_median_abs_rel_error_%": u.get("gross_income_median_abs_rel_error_%", np.nan),
            "interval_disclosure_gross_income_%": inter.get("gross_income", np.nan),
            "homogeneous_records_%": hom.get("homogeneous_records_%", np.nan),
            "dbrl_all_%": privacy.distance_based_linkage(res, "all")["linkage_rate_%"],
        })
    return pd.DataFrame(rows)


def scalability(n_values=(1000, 2000, 4000, 8000, 16000), k_values=(3, 5, 10, 20),
                fixed_k: int = 5, fixed_n: int = 8000, repeats: int = 2,
                seed: int = 42) -> pd.DataFrame:
    """
    Wall-clock time of the MDAV partition step only (encoding excluded).
    Two sweeps: n at fixed k, and k at fixed n. Median of ``repeats`` runs.
    """
    from .features import build_encoder
    cfg = AnonymizationConfig(seed=seed)
    cache: dict[int, np.ndarray] = {}

    def matrix(n: int) -> np.ndarray:
        if n not in cache:
            raw = generate_income_tax_data(n=n, seed=seed, dirty=False)
            clean, _ = validate_and_clean(raw, cfg.numeric_attributes, cfg.categorical_attributes)
            enc = build_encoder(cfg.numeric_attributes, cfg.categorical_attributes,
                                cfg.categorical_weight, True, cfg.monetary_attributes)
            cache[n] = enc.fit_transform(clean)
        return cache[n]

    def timed(X, k):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mdav(X, k)
            times.append(time.perf_counter() - t0)
        return float(np.median(times))

    rows = []
    for n in n_values:
        X = matrix(n)
        rows.append({"sweep": "n", "n": n, "k": fixed_k, "d": X.shape[1], "seconds": timed(X, fixed_k)})
    X = matrix(fixed_n)
    for k in k_values:
        rows.append({"sweep": "k", "n": fixed_n, "k": k, "d": X.shape[1], "seconds": timed(X, k)})
    return pd.DataFrame(rows)


def loglog_slope(x, y) -> float:
    """Least-squares slope of log(y) on log(x) - the empirical complexity exponent."""
    return float(np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0])
