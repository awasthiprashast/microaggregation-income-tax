"""
Reproduce every experiment, figure and table of the project.

    python run_experiments.py                 # full run (n = 5000, seed 42)
    python run_experiments.py --quick         # small smoke run
    python run_experiments.py --no-scalability

Outputs
-------
data/synthetic_income_tax.csv          raw synthetic dataset (with injected dirt)
results/summary.csv / summary.json     all metrics for every (method, k)
results/validation_report.csv          what cleaning changed (counts only)
results/ablation.csv                   distance-space ablation (MDAV, k=5)
results/collusion.csv                  background-knowledge attack curves
results/scalability.csv                MDAV runtime vs n and vs k
results/attribute_mdav_k5.csv          per-attribute statistics (aggregates)
results/regression_mdav_k5.csv         regression coefficients and CI overlap
results/record_comparison_mdav_k5.csv  sample records: BANDED original vs released
results/released/mdav_k{k}.csv         guard-checked anonymised releases
results/figures/*.png                  figures used in the report
docs/REPORT.md                         report rendered from docs/report_template.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from microagg import experiments, plotting, privacy, release, safe_display, utility
from microagg.config import AnonymizationConfig, EXPERIMENT_K_VALUES, RANDOM_SEED
from microagg.data_generator import generate_income_tax_data
from microagg.preprocessing import validate_and_clean
from microagg.report_builder import build_report

ROOT = Path(__file__).resolve().parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="number of synthetic taxpayers")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--k", type=int, nargs="+", default=list(EXPERIMENT_K_VALUES))
    ap.add_argument("--quick", action="store_true", help="n=800, no scalability benchmark")
    ap.add_argument("--no-scalability", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args(argv)
    if args.quick:
        args.n, args.no_scalability = 800, True

    out = args.out
    fig_dir, rel_dir = out / "figures", out / "released"
    for d in (out, fig_dir, rel_dir, ROOT / "data"):
        d.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. data ---------------------------------------------------------------- #
    print(f"[1/7] Generating synthetic dataset (n={args.n}, seed={args.seed})")
    raw = generate_income_tax_data(n=args.n, seed=args.seed)
    raw.to_csv(ROOT / "data" / "synthetic_income_tax.csv", index=False)

    base = AnonymizationConfig(seed=args.seed)
    clean, report = validate_and_clean(raw, base.numeric_attributes, base.categorical_attributes)
    report.to_frame().to_csv(out / "validation_report.csv", index=False)
    print("      cleaning:", report.summary())

    # 2. experiment grid ------------------------------------------------------- #
    print(f"[2/7] Running grid: methods x k={args.k}")
    summary, results = experiments.run_grid(
        clean, k_values=args.k, base=base,
        progress=lambda m, k: print(f"      done {m:<20s} k={k}"))
    summary.to_csv(out / "summary.csv", index=False)
    summary.to_json(out / "summary.json", orient="records", indent=2)

    # 3. detailed tables for k = 5 (or the closest k) ------------------------- #
    k_detail = min(args.k, key=lambda k: abs(k - 5))
    res5 = results[("mdav", k_detail)]
    utility.attribute_table(res5).to_csv(out / "attribute_mdav_k5.csv", index=False)
    reg = utility.regression_validity(res5)
    if reg:
        reg["table"].reset_index(names="term").to_csv(out / "regression_mdav_k5.csv", index=False)
    cmp = safe_display.comparison_table(res5, n=8, seed=args.seed, reveal_original=False)
    keep = [c for c in cmp.columns if c.split(" · ")[0] in
            ("age", "gender", "occupation", "gross_income", "taxable_income", "tax_paid")] + ["group size"]
    cmp[keep].reset_index(names="record").to_csv(out / "record_comparison_mdav_k5.csv", index=False)

    # 4. attacks & ablation ----------------------------------------------------- #
    print("[3/7] Background-knowledge attack curves and distance-space ablation")
    curves = experiments.collusion_curves(results)
    curves.to_csv(out / "collusion.csv", index=False)
    experiments.ablation(clean, k=k_detail, base=base).to_csv(out / "ablation.csv", index=False)

    # 5. releases (guarded) ------------------------------------------------------ #
    print("[4/7] Building guarded releases")
    guard_log = {}
    for (method, k), res in results.items():
        try:
            rel, checks = release.build_release(res)
        except release.ReleaseValidationError as exc:
            guard_log[f"{method}_k{k}"] = f"BLOCKED: {exc}"
            continue
        guard_log[f"{method}_k{k}"] = "released"
        if method == "mdav":
            rel.to_csv(rel_dir / f"mdav_k{k}.csv", index=False)
            (rel_dir / f"mdav_k{k}_metadata.json").write_text(
                json.dumps(release.release_metadata(res), indent=2), encoding="utf-8")
    (out / "release_guard_log.json").write_text(json.dumps(guard_log, indent=2), encoding="utf-8")

    # 6. scalability --------------------------------------------------------------- #
    slope_n = slope_k = float("nan")
    if not args.no_scalability:
        print("[5/7] Scalability benchmark (this takes a minute)")
        scal = experiments.scalability(seed=args.seed)
        scal.to_csv(out / "scalability.csv", index=False)
        sn, sk = scal[scal.sweep == "n"], scal[scal.sweep == "k"]
        slope_n = experiments.loglog_slope(sn["n"], sn["seconds"])
        slope_k = experiments.loglog_slope(sk["k"], sk["seconds"])
        plotting.plot_runtime(scal, fig_dir / "runtime.png", slope_n, slope_k)
    else:
        print("[5/7] Scalability benchmark skipped")

    # 7. figures ------------------------------------------------------------------ #
    print("[6/7] Figures")
    plotting.plot_metric_vs_k(summary, [("il_percent", "Information loss, raw scale (%)"),
                                        ("il_log_percent", "Information loss, log scale (%)")],
                              fig_dir / "information_loss.png")
    plotting.plot_metric_vs_k(summary, [("dbrl_all_%", "Linkage success, attacker knows all (%)"),
                                        ("dbrl_qi_%", "Linkage success, attacker knows QIs (%)")],
                              fig_dir / "linkage_risk.png", bound_one_over_k=True)
    plotting.plot_metric_vs_k(summary, [("interval_disclosure_gross_income_%", "Income disclosed within ±10% (%)"),
                                        ("homogeneous_records_%", "Records in bracket-homogeneous groups (%)")],
                              fig_dir / "attribute_disclosure.png")
    plotting.plot_tradeoff(summary, fig_dir / "tradeoff.png")
    plotting.plot_tradeoff(summary, fig_dir / "tradeoff_attribute.png",
                           y="interval_disclosure_gross_income_%",
                           ylabel="Income disclosed within ±10% (%) - lower is better privacy")
    plotting.plot_income_distribution(results, fig_dir / "income_distribution.png")
    plotting.plot_quintile_tax(results, fig_dir / "quintile_tax.png")
    if not curves.empty:
        plotting.plot_collusion(curves, fig_dir / "collusion.png")

    # run info + report ----------------------------------------------------------- #
    top1 = clean["gross_income"] >= clean["gross_income"].quantile(0.99)
    z = (clean["gross_income"] - clean["gross_income"].mean()) / clean["gross_income"].std(ddof=0)
    s = report.summary()
    info = {
        "n_records": int(len(clean)), "n_raw_rows": int(len(raw)), "seed": args.seed,
        "k_values": ", ".join(str(k) for k in args.k), "k_detail": k_detail,
        "duplicates_removed": s["duplicate_rows_removed"] + s["duplicate_ids_removed"],
        "cells_imputed": s["cells_imputed"], "cells_repaired": s["cells_repaired_by_identity"],
        "rows_dropped": s["rows_dropped_too_sparse"],
        "n_features": int(res5.X.shape[1]),
        "top1_share_of_sst": float(100 * (z[top1] ** 2).sum() / (z ** 2).sum()),
        "slope_n": slope_n, "slope_k": slope_k,
        "gender_other_original": int((clean["gender"] == "Other").sum()) if "gender" in clean else 0,
        **{f"gender_other_k{kk}": int((res.anonymized["gender"] == "Other").sum())
           for (m, kk), res in results.items() if m == "mdav" and "gender" in res.anonymized},
        "guard_blocked": ", ".join(k for k, v in guard_log.items() if v.startswith("BLOCKED")) or "none",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "total_runtime_s": time.perf_counter() - t_start,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    template = ROOT / "docs" / "report_template.md"
    if template.exists() and not args.quick and set(EXPERIMENT_K_VALUES) <= set(args.k):
        print("[7/7] Rendering docs/REPORT.md")
        build_report(template, out, ROOT / "docs" / "REPORT.md")
    else:
        print(f"[7/7] Report rendering skipped (needs a full run including k = {list(EXPERIMENT_K_VALUES)})")
    print(f"Done in {time.perf_counter() - t_start:.1f}s. Results in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
