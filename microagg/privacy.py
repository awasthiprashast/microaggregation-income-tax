"""
Disclosure-risk metrics and attack simulations.

Micro-aggregation with a JOINT partition guarantees k-anonymity on the
released attributes: each released record is indistinguishable from >= k-1
others. That bounds *identity* disclosure, but not *attribute* disclosure.
The functions below measure both, under explicit attacker models.

Identity disclosure
    k_anonymity_level          - size of the smallest equivalence class.
    distance_based_linkage     - DBRL (Pagliuca & Seri 1999; Domingo-Ferrer &
                                 Torra 2001): the attacker holds the ORIGINAL
                                 values of every target (worst case) and links
                                 each one to the nearest released record. If
                                 m released records are identical, the attacker
                                 picks one at random, so success = 1/m.

Attribute disclosure
    interval_disclosure        - share of records whose released value lies
                                 within +-p% of the true value.
    homogeneity                - share of records whose group is homogeneous in
                                 the (original) tax bracket: identify the group
                                 and you learn the bracket with certainty.
                                 Also distinct l-diversity and ordered-distance
                                 t-closeness of the bracket per group.
    dominance_rules            - (1, 50%) dominance and p%-rule on gross income
                                 per group, the rules tax agencies use for
                                 tabular data: one dominant contributor's income
                                 can be estimated from the group mean.
    attribute_inference_attack - attacker knows a target's quasi-identifiers
                                 and reads the income of the closest released
                                 record; compared with a no-release baseline.

Background knowledge
    collusion_attack           - attacker already knows the true income of a
                                 random share f of the population. If they know
                                 every other member of the target's group, the
                                 target's income follows exactly:
                                     x_t = |G| * mean(G) - sum_{j in G, j != t} x_j
                                 Expected success ~ f^(|G|-1), i.e. it decays
                                 exponentially in k but is never zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .anonymizer import AnonymizationResult
from .config import CATEGORICAL_QUASI_IDENTIFIERS, NUMERIC_QUASI_IDENTIFIERS
from .features import build_encoder
from .tax import BRACKET_LABELS


# --------------------------------------------------------------------------- #
# Identity disclosure
# --------------------------------------------------------------------------- #
def k_anonymity_level(df: pd.DataFrame, attributes) -> int:
    """Smallest number of records sharing one combination of ``attributes``."""
    attributes = list(attributes)
    if not attributes:
        return len(df)
    return int(df.groupby(attributes, dropna=False).size().min())


def _nearest_unique(Q: np.ndarray, U: np.ndarray, chunk: int = 1024) -> np.ndarray:
    """Index of the nearest row of U for every row of Q (squared Euclidean, chunked)."""
    u_sq = np.einsum("ij,ij->i", U, U)
    out = np.empty(Q.shape[0], dtype=np.int64)
    for start in range(0, Q.shape[0], chunk):
        q = Q[start:start + chunk]
        d = u_sq[None, :] - 2.0 * q @ U.T          # ||q||^2 is constant per row
        out[start:start + chunk] = np.argmin(d, axis=1)
    return out


def _attacker_attributes(result: AnonymizationResult, scope: str):
    cfg = result.config
    if scope == "all":
        return list(cfg.numeric_attributes), list(cfg.categorical_attributes)
    num = [c for c in cfg.numeric_attributes if c in NUMERIC_QUASI_IDENTIFIERS]
    cat = [c for c in cfg.categorical_attributes if c in CATEGORICAL_QUASI_IDENTIFIERS]
    return num, cat


def distance_based_linkage(result: AnonymizationResult, scope: str = "all") -> dict[str, float]:
    """
    Expected re-identification rate of distance-based record linkage.

    scope = "all": attacker knows all released attributes of the targets
                   (e.g. a leaked copy of the original file) - worst case;
    scope = "qi" : attacker knows only the quasi-identifiers (age, dependents,
                   gender, marital status, education, occupation, region, tier).
    """
    num, cat = _attacker_attributes(result, scope)
    if not num and not cat:
        return {"linkage_rate_%": float("nan"), "max_record_risk": float("nan")}
    cfg = result.config
    enc = build_encoder(num, cat, cfg.categorical_weight, cfg.log_transform_monetary,
                        cfg.monetary_attributes).fit(result.original)
    Q = enc.transform(result.original)
    R = np.round(enc.transform(result.anonymized), 9)
    U, inverse, counts = np.unique(R, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    nearest = _nearest_unique(Q, U)
    success = np.where(nearest == inverse, 1.0 / counts[inverse], 0.0)
    return {
        "linkage_rate_%": float(100 * success.mean()),
        "max_record_risk": float(success.max()),
    }


# --------------------------------------------------------------------------- #
# Attribute disclosure
# --------------------------------------------------------------------------- #
def interval_disclosure(result: AnonymizationResult, attributes=None, p: float = 0.10) -> dict[str, float]:
    """Share (%) of records whose released value is within +-p of the true value."""
    if attributes is None:
        attributes = [c for c in ("gross_income", "taxable_income", "tax_paid")
                      if c in result.config.numeric_attributes]
    out = {}
    for c in attributes:
        o = result.original[c].to_numpy(dtype=float)
        a = result.anonymized[c].to_numpy(dtype=float)
        hit = np.where(o == 0, np.isclose(a, 0.0), np.abs(a - o) <= p * np.abs(o))
        out[c] = float(100 * hit.mean())
    return out


def _labels_for(result: AnonymizationResult, attribute: str) -> np.ndarray | None:
    if result.labels is not None:
        return result.labels
    if result.attribute_labels and attribute in result.attribute_labels:
        return result.attribute_labels[attribute]
    return None


def homogeneity(result: AnonymizationResult) -> dict[str, float] | None:
    """Tax-bracket homogeneity, distinct l-diversity and t-closeness of groups."""
    if result.labels is None or "tax_bracket" not in result.original:
        return None
    brackets = pd.Categorical(result.original["tax_bracket"], categories=list(BRACKET_LABELS),
                              ordered=True)
    tab = pd.crosstab(result.labels, brackets, dropna=False)
    tab = tab.reindex(columns=list(BRACKET_LABELS), fill_value=0)
    sizes = tab.sum(axis=1).to_numpy()
    distinct = (tab > 0).sum(axis=1).to_numpy()
    homogeneous_group = distinct == 1
    record_in_homogeneous = homogeneous_group[result.labels]

    # Ordered EMD (t-closeness for an ordinal attribute): mean |cum diff| / (m-1)
    global_p = tab.sum(axis=0).to_numpy() / tab.to_numpy().sum()
    group_p = tab.to_numpy() / sizes[:, None]
    m = len(BRACKET_LABELS)
    emd = np.abs(np.cumsum(group_p - global_p, axis=1)).sum(axis=1) / (m - 1)
    return {
        "homogeneous_records_%": float(100 * record_in_homogeneous.mean()),
        "homogeneous_groups_%": float(100 * homogeneous_group.mean()),
        "min_l_diversity": int(distinct.min()),
        "mean_l_diversity": float(distinct.mean()),
        "max_t_closeness": float(emd.max()),
        "mean_t_closeness": float(emd.mean()),
    }


def dominance_rules(result: AnonymizationResult, attribute: str = "gross_income",
                    dominance_share: float = 0.5, p_percent: float = 10.0) -> dict[str, float] | None:
    """
    (1, 50%) dominance: largest contributor > 50% of the group total.
    p%-rule: the second-largest contributor can estimate the largest within p%,
             i.e. total - x1 - x2 < (p/100) * x1.
    Reported as share of groups and of records in sensitive groups.
    """
    labels = _labels_for(result, attribute)
    if labels is None or attribute not in result.original:
        return None
    x = result.original[attribute].to_numpy(dtype=float)
    size = np.bincount(labels)
    total = np.bincount(labels, weights=x)
    order = np.lexsort((-x, labels))                 # by group, then value descending
    start = np.concatenate([[0], np.cumsum(size)[:-1]])
    sorted_x = x[order]
    x1 = sorted_x[start]
    x2 = np.where(size > 1, sorted_x[np.minimum(start + 1, len(x) - 1)], 0.0)
    dominant = x1 > dominance_share * total
    p_rule = (total - x1 - x2) < (p_percent / 100.0) * x1
    return {
        "dominance_groups_%": float(100 * dominant.mean()),
        "dominance_records_%": float(100 * size[dominant].sum() / size.sum()),
        "p_rule_groups_%": float(100 * p_rule.mean()),
        "p_rule_records_%": float(100 * size[p_rule].sum() / size.sum()),
    }


def attribute_inference_attack(result: AnonymizationResult, target: str = "gross_income",
                               tolerance: float = 0.10) -> dict[str, float] | None:
    """
    Background-knowledge attack on QIs: the attacker knows the target's true
    quasi-identifiers, links to the nearest released record on those QIs and
    reads off its ``target`` value. Success = estimate within +-tolerance.
    Baseline attacker (no release): guesses the population median.
    """
    num, cat = _attacker_attributes(result, "qi")
    if target not in result.config.numeric_attributes or (not num and not cat):
        return None
    cfg = result.config
    enc = build_encoder(num, cat, cfg.categorical_weight, False, ()).fit(result.original)
    Q = enc.transform(result.original)
    R = enc.transform(result.anonymized)
    # deduplicate released QI vectors; for duplicates, the attacker averages their targets
    Rr = np.round(R, 9)
    U, inverse, counts = np.unique(Rr, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    target_released = result.anonymized[target].to_numpy(dtype=float)
    mean_target = np.bincount(inverse, weights=target_released) / counts
    estimate = mean_target[_nearest_unique(Q, U)]
    truth = result.original[target].to_numpy(dtype=float)

    def hit_rate(est):
        ok = np.where(truth == 0, np.isclose(est, 0.0), np.abs(est - truth) <= tolerance * np.abs(truth))
        return float(100 * ok.mean())

    with_release = hit_rate(estimate)
    baseline = hit_rate(np.full_like(truth, np.median(truth)))
    return {
        "inference_success_%": with_release,
        "baseline_success_%": baseline,
        "disclosure_gain_pp": with_release - baseline,
    }


def collusion_attack(result: AnonymizationResult, fractions=(0.25, 0.5, 0.75, 0.9),
                     attribute: str = "gross_income", seed: int | None = None) -> pd.DataFrame | None:
    """
    Simulate an attacker who knows the exact ``attribute`` value of a random
    share f of all records (payroll leak, colleagues, public officials...).
    A target outside the known set is fully disclosed when all other members
    of its group are known. Returns empirical and expected success rates.
    The recovered value is checked arithmetically against the truth.
    """
    labels = _labels_for(result, attribute)
    if labels is None or attribute not in result.original:
        return None
    rng = np.random.default_rng(result.config.seed if seed is None else seed)
    n = len(labels)
    sizes = np.bincount(labels)
    released = result.anonymized[attribute].to_numpy(dtype=float)
    truth = result.original[attribute].to_numpy(dtype=float)
    group_sum_released = np.bincount(labels, weights=released)  # = |G| * mean(G)
    rows = []
    for f in fractions:
        known = rng.random(n) < f
        unknown_per_group = np.bincount(labels, weights=(~known).astype(float))
        targets = ~known
        if not targets.any():
            continue
        # target is exposed iff it is the only unknown member of its group
        exposed = targets & (unknown_per_group[labels] == 1)
        known_sum = np.bincount(labels, weights=np.where(known, truth, 0.0))
        recovered = group_sum_released[labels] - known_sum[labels]
        exact = exposed & np.isclose(recovered, truth, rtol=1e-6, atol=1.0)
        expected = np.mean(f ** (sizes[labels[targets]] - 1))
        rows.append({
            "known_fraction": f,
            "exposed_targets_%": float(100 * exposed[targets].mean()),
            "exactly_recovered_%": float(100 * exact[targets].mean()),
            "expected_%": float(100 * expected),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def compute_privacy(result: AnonymizationResult) -> dict[str, float]:
    """Flat dict of scalar privacy metrics for one anonymisation run."""
    cfg = result.config
    out: dict[str, float] = {
        "k_anonymity_level": k_anonymity_level(result.anonymized, cfg.released_attributes),
    }
    out["k_anonymity_satisfied"] = float(out["k_anonymity_level"] >= cfg.k)
    for scope in ("all", "qi"):
        d = distance_based_linkage(result, scope)
        out[f"dbrl_{scope}_%"] = d["linkage_rate_%"]
        out[f"dbrl_{scope}_max_record_risk"] = d["max_record_risk"]
    for attr, v in interval_disclosure(result).items():
        out[f"interval_disclosure_{attr}_%"] = v
    for block in (homogeneity(result), dominance_rules(result), attribute_inference_attack(result)):
        if block:
            out.update(block)
    coll = collusion_attack(result, fractions=(0.5, 0.9))
    if coll is not None:
        for _, r in coll.iterrows():
            out[f"collusion_f{r['known_fraction']:.1f}_%"] = r["exactly_recovered_%"]
    return out
