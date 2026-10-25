"""
Real-data experiments (US Census ACS 2022 PUMS, California).

    python run_real_experiments.py            # full run (~20 min)
    python run_real_experiments.py --quick    # k=5 only, no seeds / sweeps

Writes results/real/*.csv and results/real/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from microagg import advanced as A
from microagg import realdata as R

OUT = Path(__file__).resolve().parent / "results" / "real"
FIG = OUT / "figures"
LABELS = A.METHODS
COLORS = {"mdav_all": "#1f77b4", "mdav_qi": "#2ca02c", "da_mdav": "#d62728", "random": "#7f7f7f"}


def run_grid(df, ref, ks, methods, seed, **kw) -> list[dict]:
    rows = []
    for m in methods:
        for k in ks:
            t0 = time.perf_counter()
            rel = A.anonymize(df, m, k, seed=seed, **kw)
            row = A.evaluate(df, ref, rel, seed)
            row["runtime_s"] = time.perf_counter() - t0
            rows.append(row)
            print(f"  {m:9s} k={k:<2d} il={row['il_%']:.1f} link={row['linkage_%']:.1f} "
                  f"qi_link_band={row['qi_link_band_acc_%']:.1f} ({row['runtime_s']:.0f}s)", flush=True)
    return rows


def figures(summary: pd.DataFrame, baseline: dict) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    base = baseline["band_acc_%"]
    plain = summary[summary["noise_eps"].isna()]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, g in plain.groupby("method"):
        g = g.sort_values("k")
        ax[0].plot(g["il_%"], g["qi_link_band_acc_%"], "o-", color=COLORS[m], label=LABELS[m])
        for _, r in g.iterrows():
            ax[0].annotate(f"k={int(r['k'])}", (r["il_%"], r["qi_link_band_acc_%"]),
                           textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax[1].plot(g["k"], g["linkage_%"], "o-", color=COLORS[m], label=LABELS[m])
    ax[0].axhline(base, ls="--", color="k", lw=1)
    ax[0].text(ax[0].get_xlim()[0], base + 0.6, "no-release baseline", fontsize=8)
    ax[0].set(xlabel="Information loss (%)", ylabel="Income-band inference accuracy (%)",
              title="Utility vs. attribute disclosure")
    ax[1].set(xlabel="k", ylabel="Record-linkage success (%)", title="Identity disclosure", yscale="log")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "tradeoff.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, g in plain.groupby("method"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["qi_link_band_acc_%"] - base, "o-", color=COLORS[m], label=LABELS[m])
    ax.axhline(0, ls="--", color="k", lw=1)
    ax.set(xlabel="k", ylabel="Excess over no-release baseline (pp)",
           title="Attribute disclosure caused by the release")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "excess_disclosure.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for m, g in plain.groupby("method"):
        g = g.sort_values("k")
        ax.plot(g["k"], g["reg_median_coef_err_%"], "o-", color=COLORS[m], label=LABELS[m])
    ax.set(xlabel="k", ylabel="Median regression-coefficient error (%)",
           title="Analytical validity (log-income regression)", yscale="log")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "regression_error.png", dpi=160); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 10, 20])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df, ref = R.load_samples()
    df, ref = R.add_income_band(df), R.add_income_band(ref)
    ks = [5] if args.quick else args.ks
    methods = ["mdav_all", "mdav_qi", "da_mdav", "random"]

    base = A.ml_attack(df, None, ref)
    baseline = {"band_acc_%": base["ml_band_acc_%"], "within25_%": base["ml_within25_%"],
                "majority_class_%": 100 * df["income_band"].value_counts(normalize=True).max()}
    (OUT / "baseline.json").write_text(json.dumps(baseline, indent=2))
    print("no-release baseline:", baseline)

    print("Main grid"); rows = run_grid(df, ref, ks, methods, 42)
    summary = pd.DataFrame(rows)

    if not args.quick:
        print("Noisy centroids (k=5)")
        noisy = []
        for eps in (1.0, 5.0):
            for m in ("mdav_all", "da_mdav"):
                noisy += run_grid(df, ref, [5], [m], 42, noise_eps=eps)
        summary = pd.concat([summary, pd.DataFrame(noisy)], ignore_index=True)

        print("Constraint sweep (DA-MDAV, k=5)")
        sweep = []
        for l, t in [(2, 0.40), (3, 0.30), (3, 0.25), (4, 0.20)]:
            for r in run_grid(df, ref, [5], ["da_mdav"], 42, l=l, t=t):
                r.update(l=l, t=t); sweep.append(r)
        pd.DataFrame(sweep).to_csv(OUT / "constraint_sweep.csv", index=False)

        print("Robustness over 3 independent samples (k=5)")
        rob = []
        for s in (1, 2, 3):
            a, b = R.build_samples(seed=s, save=False)
            a, b = R.add_income_band(a), R.add_income_band(b)
            for r in run_grid(a, b, [5], methods, s):
                r["sample_seed"] = s; rob.append(r)
        pd.DataFrame(rob).to_csv(OUT / "robustness.csv", index=False)

    summary.to_csv(OUT / "summary.csv", index=False)
    figures(summary, baseline)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
