"""
Static figures for the report (matplotlib, PNG).

Colour roles (validated categorical palette, slots 1-3 are CVD-safe all-pairs):
    MDAV = blue, individual ranking = orange, random partition = aqua.
k is ordinal, so k-indexed series use one blue ramp light -> dark.
Original-data references are neutral grey.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import METHOD_LABELS  # noqa: E402

METHOD_COLORS = {"mdav": "#2a78d6", "individual_ranking": "#eb6834", "random": "#1baf7a"}
METHOD_MARKERS = {"mdav": "o", "individual_ranking": "s", "random": "^"}
K_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NEUTRAL = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK_2, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#ecebe7",
        "grid.linewidth": 0.8, "xtick.color": INK_2, "ytick.color": INK_2,
        "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 7,
    })


def _k_colors(ks):
    ks = sorted(ks)
    ramp = K_RAMP if len(ks) <= len(K_RAMP) else plt.cm.Blues(np.linspace(0.35, 0.95, len(ks)))
    return {k: ramp[i] for i, k in enumerate(ks)}


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_metric_vs_k(summary: pd.DataFrame, metrics: list[tuple[str, str]], path: Path,
                     bound_one_over_k: bool = False):
    """Small multiples: one panel per (column, title), one line per method."""
    _style()
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 3.6), squeeze=False)
    for ax, (col, title) in zip(axes[0], metrics):
        for m, g in summary.groupby("method", sort=False):
            g = g.sort_values("k")
            if g[col].isna().all():
                continue
            ax.plot(g["k"], g[col], marker=METHOD_MARKERS[m], color=METHOD_COLORS[m],
                    label=METHOD_LABELS[m], markeredgecolor="white", markeredgewidth=1.5)
        if bound_one_over_k and "dbrl" in col:
            ks = np.array(sorted(summary["k"].unique()))
            ax.plot(ks, 100.0 / ks, ls="--", lw=1.2, color=NEUTRAL, label="k-anonymity bound 100/k")
        ax.set_title(title, loc="left")
        ax.set_xlabel("k (minimum group size)")
        ax.set_xticks(sorted(summary["k"].unique()))
        ax.set_ylim(bottom=0)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in labels:
                handles.append(hh)
                labels.append(ll)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.08))
    _save(fig, path)


def plot_tradeoff(summary: pd.DataFrame, path: Path, x="il_percent", y="dbrl_all_%",
                  xlabel="Information loss IL (%) - lower is better utility",
                  ylabel="Record-linkage success (%) - lower is better privacy"):
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for m, g in summary.groupby("method", sort=False):
        g = g.sort_values("k")
        ax.plot(g[x], g[y], marker=METHOD_MARKERS[m], color=METHOD_COLORS[m], label=METHOD_LABELS[m],
                markeredgecolor="white", markeredgewidth=1.5, markersize=8)
        for i, (_, r) in enumerate(g.iterrows()):
            # alternate above/below so labels of nearby points do not collide
            ax.annotate(f"k={int(r['k'])}", (r[x], r[y]), textcoords="offset points",
                        xytext=(5, 7 if i % 2 == 0 else -13), fontsize=8, color=INK_2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Privacy–utility trade-off", loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    _save(fig, path)


def plot_income_distribution(results: dict, path: Path, method="mdav"):
    """Smoothed log10 gross-income density: original (grey fill) vs anonymised for
    each k. Only the original P1-P99 range is drawn, so no individual outlier
    can be read off the axis."""
    from scipy.stats import gaussian_kde
    _style()
    ks = sorted(k for (m, k) in results if m == method)
    any_res = results[(method, ks[0])]
    orig = any_res.original["gross_income"].astype(float)
    lo, cap = orig.quantile(0.01), orig.quantile(0.99)       # trimmed to P1-P99 for display
    grid = np.linspace(np.log10(lo), np.log10(cap), 300)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    def density(values):
        v = np.log10(values[(values >= lo) & (values <= cap)])
        return gaussian_kde(v, bw_method=0.12)(grid)

    ax.fill_between(grid, density(orig), color="#dcdbd6", lw=0, label="Original")
    colors = _k_colors(ks)
    for k in ks:
        a = results[(method, k)].anonymized["gross_income"].astype(float)
        ax.plot(grid, density(a), lw=1.8, color=colors[k], label=f"MDAV k={k}")
    ax.text(0.99, 0.02, "range shown: original P1–P99", transform=ax.transAxes, ha="right",
            fontsize=8, color=INK_2)
    ticks = [2e5, 5e5, 1e6, 2e6, 5e6]
    ax.set_xticks(np.log10(ticks))
    ax.set_xticklabels(["2L", "5L", "10L", "20L", "50L"])
    ax.set_xlabel("Gross income (log scale, ₹)")
    ax.set_ylabel("Density")
    ax.set_title("Gross-income distribution before and after micro-aggregation", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_collusion(curves: pd.DataFrame, path: Path):
    _style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = _k_colors(curves["k"].unique())
    for k, g in curves.groupby("k"):
        ax.plot(g["known_fraction"] * 100, g["exactly_recovered_%"], marker="o", color=colors[k],
                label=f"k={k} (simulated)", markeredgecolor="white", markeredgewidth=1.5)
        ax.plot(g["known_fraction"] * 100, g["expected_%"], ls="--", lw=1.1, color=colors[k])
    ax.plot([], [], ls="--", lw=1.1, color=NEUTRAL, label="expected ≈ f^(|G|-1)")
    ax.set_xlabel("Share of population whose true income the attacker already knows (%)")
    ax.set_ylabel("Targets whose income is recovered exactly (%)")
    ax.set_title("Background-knowledge (collusion) attack on MDAV releases", loc="left")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_runtime(scal: pd.DataFrame, path: Path, slope_n: float, slope_k: float):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    sn = scal[scal["sweep"] == "n"]
    ax = axes[0]
    ax.loglog(sn["n"], sn["seconds"], marker="o", color=METHOD_COLORS["mdav"],
              markeredgecolor="white", markeredgewidth=1.5, label="measured")
    ref = sn["seconds"].iloc[0] * (sn["n"] / sn["n"].iloc[0]) ** 2
    ax.loglog(sn["n"], ref, ls="--", lw=1.2, color=NEUTRAL, label="∝ n² reference")
    ax.set_title(f"Runtime vs n (k={int(sn['k'].iloc[0])}): slope {slope_n:.2f}", loc="left")
    ax.set_xlabel("records n")
    ax.set_ylabel("seconds")
    ax.legend()
    sk = scal[scal["sweep"] == "k"]
    ax = axes[1]
    ax.loglog(sk["k"], sk["seconds"], marker="o", color=METHOD_COLORS["mdav"],
              markeredgecolor="white", markeredgewidth=1.5, label="measured")
    ref = sk["seconds"].iloc[0] * (sk["k"].iloc[0] / sk["k"])
    ax.loglog(sk["k"], ref, ls="--", lw=1.2, color=NEUTRAL, label="∝ 1/k reference")
    ax.set_title(f"Runtime vs k (n={int(sk['n'].iloc[0])}): slope {slope_k:.2f}", loc="left")
    ax.set_xlabel("k")
    ax.set_xticks(sk["k"])
    ax.set_xticklabels([str(int(k)) for k in sk["k"]])
    ax.legend()
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    for a in axes:
        a.xaxis.set_minor_formatter(NullFormatter())
        a.yaxis.set_minor_formatter(NullFormatter())
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    _save(fig, path)


def plot_quintile_tax(results: dict, path: Path, method="mdav"):
    from .utility import effective_tax_by_quintile
    _style()
    ks = sorted(k for (m, k) in results if m == method)
    base = effective_tax_by_quintile(results[(method, ks[0])].original)
    series = {"Original": base}
    for k in ks:
        series[f"MDAV k={k}"] = effective_tax_by_quintile(results[(method, k)].anonymized)
    df = pd.DataFrame(series)
    colors = ["#b5b4ae"] + [_k_colors(ks)[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.8 / df.shape[1]
    x = np.arange(len(df))
    for i, (col, c) in enumerate(zip(df.columns, colors)):
        ax.bar(x + (i - (df.shape[1] - 1) / 2) * width, df[col], width=width * 0.9, color=c, label=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q}\n(poorest)" if q == "Q1" else (f"{q}\n(richest)" if q == "Q5" else q)
                        for q in df.index])
    ax.set_ylabel("Effective tax rate (%)")
    ax.set_title("Effective tax rate by gross-income quintile", loc="left")
    ax.legend(fontsize=8, ncol=3)
    _save(fig, path)
