"""
Contributions beyond plain MDAV (evaluated on real ACS data).

1. QI-space MDAV ("mdav_qi")
   Standard MDAV clusters on *all* attributes, including the confidential
   incomes. Records with similar income therefore land in the same group, so the
   group mean is close to every member's true income (attribute disclosure).
   mdav_qi builds the partition on the quasi-identifiers only and aggregates
   every released attribute with that partition: members of a group look alike
   to an attacker but have *different* incomes.

2. Diversity-aware MDAV ("da_mdav")
   mdav_qi followed by a constraint-repair step that merges groups until each
   group satisfies l-diversity (>= l distinct income bands) and t-closeness
   (earth mover's distance between the group's income-band distribution and the
   population's is <= t). k-anonymity is preserved because groups only grow.

3. Noisy centroids ("noise_eps")
   Laplace noise calibrated to the sensitivity of a clipped group mean,
   (hi - lo) / group_size, split over the confidential attributes. This is
   DP-*inspired*: the partition itself depends on the data, so the result is not a
   formal epsilon-DP guarantee (see Soria-Comas et al. 2014 for insensitive
   microaggregation, which makes it formal).

4. Machine-learning reconstruction attack
   A random forest is trained on the released data and asked to predict each
   original person's income band from their quasi-identifiers. The same model
   trained on a disjoint *reference* sample (public population knowledge, no
   release) gives the baseline. Only the *excess* over that baseline is
   disclosure caused by the release.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import mdav as algorithms
from .features import build_encoder
from .realdata import CATEGORICAL_QI, CONFIDENTIAL, MONETARY, NUMERIC_QI


# --------------------------------------------------------------------------- #
# Distance spaces
# --------------------------------------------------------------------------- #
def encode(df: pd.DataFrame, space: str, weight: float = 1.0 / np.sqrt(2.0)):
    """Return (X, encoder) for the 'qi' space or the 'all' space."""
    if space == "qi":
        num, cat = NUMERIC_QI, CATEGORICAL_QI
    else:
        num, cat = NUMERIC_QI + CONFIDENTIAL, CATEGORICAL_QI
    enc = build_encoder(list(num), list(cat), weight, True, MONETARY)
    return enc.fit_transform(df), enc


# --------------------------------------------------------------------------- #
# Diversity-aware repair
# --------------------------------------------------------------------------- #
def emd_ordered(counts: np.ndarray, global_p: np.ndarray) -> float:
    """Earth mover's distance between two distributions on an ordered scale (Li et al. 2007)."""
    p = counts / counts.sum()
    return float(np.abs(np.cumsum(p - global_p))[:-1].sum() / (len(p) - 1))


def _violation(c: np.ndarray, global_p: np.ndarray, l: int, t: float) -> float:
    """0 when the group satisfies l-diversity and t-closeness, else a positive score."""
    return max(0.0, l - (c > 0).sum()) + max(0.0, emd_ordered(c, global_p) - t) / t


def enforce_diversity(labels: np.ndarray, band: np.ndarray, X: np.ndarray, l: int, t: float,
                      n_candidates: int = 60, max_passes: int = 10, swap_penalty: float = 0.005,
                      ) -> np.ndarray:
    """
    Make every group l-diverse and t-close on the income band.

    Step 1 (swaps, group sizes unchanged): for each violating group look at the
    nearest groups in quasi-identifier space and swap one record of the group with
    one record of a neighbour when that reduces the total violation score; a small
    penalty on the extra distance keeps swaps local.
    Step 2 (merges): groups that are still violating are merged with the nearby
    group giving the lowest EMD. Groups only grow, so k-anonymity is preserved.
    """
    labels = labels.copy()
    m = int(band.max()) + 1
    global_p = np.bincount(band, minlength=m) / len(band)
    ids = list(np.unique(labels))
    members = {g: list(np.flatnonzero(labels == g)) for g in ids}
    counts = {g: np.bincount(band[idx], minlength=m).astype(float) for g, idx in members.items()}
    cent = {g: X[idx].mean(axis=0) for g, idx in members.items()}
    scale = np.sqrt(X.shape[1])
    viol = lambda c: _violation(c, global_p, l, t)       # noqa: E731

    for _ in range(max_passes):
        bad = [g for g in ids if viol(counts[g]) > 0]
        if not bad:
            break
        centroids = np.array([cent[g] for g in ids])
        position = {g: i for i, g in enumerate(ids)}
        swapped = 0
        for g in bad:
            if viol(counts[g]) == 0:
                continue
            d = np.linalg.norm(centroids - cent[g], axis=1)
            neigh = [ids[j] for j in np.argsort(d)[1:n_candidates + 1]]
            best, best_delta = None, -1e-9
            for h in neigh:
                before = viol(counts[g]) + viol(counts[h])
                for a in members[g]:
                    for b in members[h]:
                        if band[a] == band[b]:
                            continue
                        cg, ch = counts[g].copy(), counts[h].copy()
                        cg[band[a]] -= 1; cg[band[b]] += 1
                        ch[band[b]] -= 1; ch[band[a]] += 1
                        extra = (np.linalg.norm(X[a] - cent[h]) + np.linalg.norm(X[b] - cent[g])
                                 - np.linalg.norm(X[a] - cent[g]) - np.linalg.norm(X[b] - cent[h]))
                        delta = viol(cg) + viol(ch) - before + swap_penalty * extra / scale
                        if delta < best_delta:
                            best, best_delta = (h, a, b), delta
            if best is None:
                continue
            h, a, b = best
            members[g][members[g].index(a)] = b
            members[h][members[h].index(b)] = a
            for key in (g, h):
                counts[key] = np.bincount(band[members[key]], minlength=m).astype(float)
                cent[key] = X[members[key]].mean(axis=0)
                centroids[position[key]] = cent[key]
            swapped += 1
        if swapped == 0:
            break

    queue = [g for g in ids if viol(counts[g]) > 0]
    while queue:
        g = queue.pop(0)
        if g not in members or viol(counts[g]) == 0 or len(members) == 1:
            continue
        others = [h for h in members if h != g]
        d = np.array([np.linalg.norm(cent[g] - cent[h]) for h in others])
        best, best_score = None, np.inf
        for j in np.argsort(d)[:n_candidates]:
            h = others[j]
            score = emd_ordered(counts[g] + counts[h], global_p) + 0.02 * d[j] / scale
            if score < best_score:
                best, best_score = h, score
        merged = members[g] + members[best]
        for key in (g, best):
            members.pop(key); counts.pop(key); cent.pop(key)
        members[g] = merged
        counts[g] = np.bincount(band[merged], minlength=m).astype(float)
        cent[g] = X[merged].mean(axis=0)
        if viol(counts[g]) > 0:
            queue.append(g)

    new = np.empty_like(labels)
    for new_id, idx in enumerate(members.values()):
        new[idx] = new_id
    return new


# --------------------------------------------------------------------------- #
# Anonymisation
# --------------------------------------------------------------------------- #
METHODS = {
    "mdav_all": "MDAV (all attributes)",
    "mdav_qi": "MDAV (QI space)",
    "da_mdav": "DA-MDAV (QI space + l-diversity/t-closeness)",
    "random": "Random partition",
}


@dataclass
class Release:
    method: str
    k: int
    labels: np.ndarray
    anonymized: pd.DataFrame           # aligned with the original (INTERNAL ONLY)
    noise_eps: float | None = None


def add_centroid_noise(anon: pd.DataFrame, original: pd.DataFrame, labels: np.ndarray,
                       eps: float, seed: int) -> pd.DataFrame:
    """Replace each group's confidential means by Laplace-perturbed means of clipped values."""
    rng = np.random.default_rng(seed)
    out = anon.copy()
    sizes = np.bincount(labels)
    eps_attr = eps / len(CONFIDENTIAL)
    for c in CONFIDENTIAL:
        lo, hi = original[c].quantile([0.01, 0.99]).to_numpy()
        clipped = original[c].clip(lo, hi)
        means = clipped.groupby(labels).mean().to_numpy()
        scale = (hi - lo) / (sizes * eps_attr)
        noisy = np.clip(means + rng.laplace(0.0, scale), 0.0, None)
        out[c] = noisy[labels]
    return out


def anonymize(original: pd.DataFrame, method: str, k: int, seed: int = 42,
              l: int = 3, t: float = 0.25, noise_eps: float | None = None) -> Release:
    """Run one of the methods in METHODS on ``original`` (needs an ``income_band`` column)."""
    n = len(original)
    if method == "mdav_all":
        X, _ = encode(original, "all")
        labels = algorithms.mdav(X, k)
    elif method in ("mdav_qi", "da_mdav"):
        X, _ = encode(original, "qi")
        labels = algorithms.mdav(X, k)
        if method == "da_mdav":
            labels = enforce_diversity(labels, original["income_band"].to_numpy(), X, l, t)
    elif method == "random":
        labels = algorithms.random_partition(n, k, seed)
    else:
        raise ValueError(f"unknown method {method!r}")
    numeric = list(NUMERIC_QI + CONFIDENTIAL)
    anon = algorithms.aggregate(original, labels, numeric, list(CATEGORICAL_QI))
    if noise_eps is not None:
        anon = add_centroid_noise(anon, original, labels, noise_eps, seed)
    return Release(method, k, labels, anon, noise_eps)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def information_loss(original: pd.DataFrame, anon: pd.DataFrame) -> float:
    """SSE / SST (%) over the numeric attributes standardised to unit variance."""
    cols = list(NUMERIC_QI + CONFIDENTIAL)
    sd = original[cols].std().replace(0, 1.0)
    zo = (original[cols] - original[cols].mean()) / sd
    za = (anon[cols] - original[cols].mean()) / sd
    return float(100.0 * ((zo - za) ** 2).to_numpy().sum() / (zo ** 2).to_numpy().sum())


def linkage_rate(original: pd.DataFrame, rel: Release) -> float:
    """
    Distance-based record linkage, attacker knows ALL attributes of the target:
    link to the nearest released record; expected success counts 1/|group| when the
    nearest released group is the target's own group.
    """
    Xo, enc = encode(original, "all")
    Xr = enc.transform(rel.anonymized)
    first = pd.Series(np.arange(len(original))).groupby(rel.labels).first().to_numpy()
    U = Xr[first]
    sizes = np.bincount(rel.labels)
    hit = np.empty(len(original))
    for s in range(0, len(original), 1024):
        q = Xo[s:s + 1024]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ U.T + (U ** 2).sum(1)[None, :]
        nearest = d.argmin(axis=1)
        hit[s:s + 1024] = nearest == rel.labels[s:s + 1024]
    return float(100.0 * (hit / sizes[rel.labels]).mean())


def diversity_stats(original: pd.DataFrame, labels: np.ndarray) -> dict[str, float]:
    band = original["income_band"].to_numpy()
    m = int(band.max()) + 1
    gp = np.bincount(band, minlength=m) / len(band)
    counts = pd.crosstab(labels, band).reindex(columns=range(m), fill_value=0).to_numpy()
    distinct = (counts > 0).sum(axis=1)
    emd = np.array([emd_ordered(c.astype(float), gp) for c in counts])
    sizes = counts.sum(axis=1)
    return {
        "groups": int(len(counts)), "min_group": int(sizes.min()),
        "mean_group": float(sizes.mean()), "max_group": int(sizes.max()),
        "min_distinct_bands": int(distinct.min()),
        "share_groups_lt3_bands_%": float(100 * (distinct < 3).mean()),
        "mean_emd": float(emd.mean()), "max_emd": float(emd.max()),
        "share_records_in_homogeneous_%": float(100 * (sizes[distinct == 1]).sum() / len(labels)),
    }


def _band_of(income: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    return np.searchsorted(cuts, income, side="right")


def fidelity(original: pd.DataFrame, rel: Release) -> dict[str, float]:
    """Utility-side measure: how close is each released income to the record's own income."""
    o = original["total_income"].to_numpy()
    r = rel.anonymized["total_income"].to_numpy()
    return {"income_within10_%": float(100 * (np.abs(r - o) <= 0.10 * o).mean())}


def qi_linkage_attack(original: pd.DataFrame, rel: Release) -> dict[str, float]:
    """
    Attribute-inference attack by linkage. The attacker knows the target's
    quasi-identifiers (e.g. from a public register), finds the released group
    that is nearest in quasi-identifier space and reads its (aggregated) income.
    """
    Xo, enc = encode(original, "qi")
    Xr = enc.transform(rel.anonymized)
    first = pd.Series(np.arange(len(original))).groupby(rel.labels).first().to_numpy()
    U = Xr[first]
    group_income = rel.anonymized["total_income"].to_numpy()[first]
    nearest = np.empty(len(original), dtype=int)
    for s in range(0, len(original), 1024):
        q = Xo[s:s + 1024]
        d = (q ** 2).sum(1)[:, None] - 2 * q @ U.T + (U ** 2).sum(1)[None, :]
        nearest[s:s + 1024] = d.argmin(axis=1)
    pred = group_income[nearest]
    cuts = original.groupby("income_band")["total_income"].min().to_numpy()[1:]
    o = original["total_income"].to_numpy()
    return {"qi_link_band_acc_%": float(100 * (_band_of(pred, cuts) == original["income_band"].to_numpy()).mean()),
            "qi_link_within25_%": float(100 * (np.abs(pred - o) <= 0.25 * o).mean())}


def _qi_matrix(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    X = pd.get_dummies(df[list(NUMERIC_QI) + list(CATEGORICAL_QI)], columns=list(CATEGORICAL_QI),
                       dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def ml_attack(original: pd.DataFrame, rel: Release | None, reference: pd.DataFrame,
              seed: int = 42) -> dict[str, float]:
    """
    Random-forest reconstruction attack. Train on the released data (rel) or, when
    rel is None, on the reference sample (no-release baseline); predict the income
    band and income of the ORIGINAL people from their quasi-identifiers.
    """
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if rel is None:
        train_X, train_band = _qi_matrix(reference), reference["income_band"].to_numpy()
        train_y = np.log1p(reference["total_income"].to_numpy())
    else:
        a = rel.anonymized
        cuts = original.groupby("income_band")["total_income"].min().to_numpy()[1:]
        train_X, train_band = _qi_matrix(a), _band_of(a["total_income"].to_numpy(), cuts)
        train_y = np.log1p(a["total_income"].to_numpy())
    target_X = _qi_matrix(original, list(train_X.columns))
    clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1,
                                 random_state=seed).fit(train_X, train_band)
    reg = RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1,
                                random_state=seed).fit(train_X, train_y)
    pred_band = clf.predict(target_X)
    pred_inc = np.expm1(reg.predict(target_X))
    o = original["total_income"].to_numpy()
    return {
        "ml_band_acc_%": float(100 * (pred_band == original["income_band"].to_numpy()).mean()),
        "ml_within25_%": float(100 * (np.abs(pred_inc - o) <= 0.25 * o).mean()),
    }


def utility_metrics(original: pd.DataFrame, anon: pd.DataFrame) -> dict[str, float]:
    """Regression validity, inequality and correlation preservation."""
    def ols(df: pd.DataFrame, cols=None):
        X = pd.get_dummies(df[["education", "sex"]], dtype=float)
        X = X.drop(columns=["education_Below high school", "sex_Female"], errors="ignore")
        X["age"] = (df["age"] - 50) / 18.0
        X["age2"] = X["age"] ** 2
        X["hours"] = df["hours_per_week"] / 40.0
        X.insert(0, "const", 1.0)
        if cols is not None:
            X = X.reindex(columns=cols, fill_value=0.0)
        y = np.log1p(df["total_income"].to_numpy())
        beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
        r2 = 1 - ((y - X.to_numpy() @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        return pd.Series(beta, index=X.columns), float(r2)

    b0, r0 = ols(original)
    b1, r1 = ols(anon, list(b0.index))
    slope = [c for c in b0.index if c != "const"]
    rel_err = float((np.abs(b1[slope] - b0[slope]) / np.abs(b0[slope])).median() * 100)

    def gini(x: np.ndarray) -> float:
        x = np.sort(x.astype(float)); n = len(x)
        return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))

    num = list(NUMERIC_QI + CONFIDENTIAL)
    corr_dist = float(np.abs(original[num].corr().to_numpy() - anon[num].corr().to_numpy()).mean())
    return {
        "reg_median_coef_err_%": rel_err, "reg_r2_orig": r0, "reg_r2_anon": r1,
        "gini_orig": gini(original["total_income"].to_numpy()),
        "gini_anon": gini(anon["total_income"].to_numpy()),
        "corr_distortion": corr_dist,
        "mean_income_error_%": float(100 * abs(anon["total_income"].mean() / original["total_income"].mean() - 1)),
    }


def evaluate(original: pd.DataFrame, reference: pd.DataFrame, rel: Release,
             seed: int = 42, with_ml: bool = True) -> dict[str, float]:
    row: dict[str, float] = {"method": rel.method, "k": rel.k,
                             "noise_eps": rel.noise_eps if rel.noise_eps is not None else np.nan,
                             "il_%": information_loss(original, rel.anonymized),
                             "linkage_%": linkage_rate(original, rel)}
    row.update(diversity_stats(original, rel.labels))
    row.update(fidelity(original, rel))
    row.update(qi_linkage_attack(original, rel))
    row.update(utility_metrics(original, rel.anonymized))
    if with_ml:
        row.update(ml_attack(original, rel, reference, seed))
    return row
