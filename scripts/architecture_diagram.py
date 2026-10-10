"""
Draw the system architecture diagram.

    python scripts/architecture_diagram.py

Writes docs/architecture.png (300 dpi) and docs/architecture.svg.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, INK_2, MUTED = "#0b0b0b", "#3d3c39", "#6b6a66"
STYLES = {                         # (fill, border)
    "entry": ("#f1f0ec", "#8a8985"),
    "pipeline": ("#e3eefb", "#2a78d6"),
    "eval": ("#dff5ec", "#1baf7a"),
    "release": ("#fde9df", "#eb6834"),
    "report": ("#ecebf7", "#4a3aa7"),
    "config": ("#f6f5f1", "#8a8985"),
}


def box(ax, x, y, w, h, title, body="", kind="pipeline", title_size=10, body_size=8):
    fill, edge = STYLES[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2))
    if body:
        ax.text(x + w / 2, y + h - 1.15, title, ha="center", va="top", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)
        ax.text(x + w / 2, y + h - 3.1, body, ha="center", va="top", fontsize=body_size,
                color=INK_2, linespacing=1.35, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_size,
                fontweight="bold", color=INK, zorder=3)


def arrow(ax, start, end, color=MUTED, style="-|>", lw=1.6, ls="-", rad=0.0):
    ax.annotate("", xy=end, xytext=start, zorder=1,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(17, 11))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ax.text(1, 64.6, "System architecture — Micro-aggregation for income & tax data publication",
            fontsize=16, fontweight="bold", color=INK, va="top")
    ax.text(1, 61.9, "Python package microagg/ · data flows left → right · original data stays "
            "inside the custodian boundary; only the guarded release leaves it",
            fontsize=10, color=MUTED, va="top")

    # ---------------- entry points ---------------- #
    box(ax, 3, 52.5, 22, 6.5, "run_experiments.py", "CLI: data → experiments →\nfigures → report",
        "entry")
    box(ax, 30, 52.5, 22, 6.5, "app.py", "Streamlit dashboard\n(7 interactive tabs)", "entry")
    box(ax, 57, 52.5, 18, 6.5, "tests/", "pytest: 26 checks", "entry")
    ax.text(77, 55.75, "Entry points", fontsize=10, color=MUTED, style="italic", va="center")
    for x in (14, 41, 66):
        arrow(ax, (x, 52.5), (x, 49.6))

    # ---------------- custodian boundary ---------------- #
    ax.add_patch(Rectangle((1.5, 3), 85, 46.6, fill=False, linestyle=(0, (6, 4)), linewidth=1.4,
                           edgecolor="#9a9994", zorder=0))
    ax.text(2.6, 48.4, "DATA CUSTODIAN ENVIRONMENT  ·  original records are internal only",
            fontsize=9, color=MUTED, fontweight="bold", va="center")

    # anonymizer band (orchestrates encode → partition → aggregate)
    ax.add_patch(FancyBboxPatch((30, 43.6), 38, 2.8, boxstyle="round,pad=0,rounding_size=0.6",
                                linewidth=1.2, edgecolor="#2a78d6", facecolor="#f4f8fd", zorder=2))
    ax.text(49, 45.0, "anonymizer.py  ·  anonymize(data, config) → AnonymizationResult",
            ha="center", va="center", fontsize=9, color="#1c5cab", fontweight="bold", zorder=3)

    # ---------------- anonymisation pipeline ---------------- #
    y, h, w = 32.5, 10, 12.2
    xs = [3, 16.5, 30, 43, 56]
    box(ax, xs[0], y, w, h, "① Data source",
        "data_generator.py\n+ tax.py (synthetic)\nor uploaded CSV")
    box(ax, xs[1], y, w, h, "② Validate & clean",
        "preprocessing.py\nranges · edit rules\nrepair · imputation")
    box(ax, xs[2], y, w, h, "③ Encode",
        "features.py\nz-score · log1p(money)\none-hot categories")
    box(ax, xs[3], y, w, h, "④ Partition",
        "mdav.py\nMDAV (main algorithm)\nbaselines: IR · random")
    box(ax, xs[4], y, w, h, "⑤ Aggregate",
        "mdav.aggregate\nmean (numeric)\nmode (categorical)")
    for a, b in zip(xs[:-1], xs[1:]):
        arrow(ax, (a + w, y + h / 2), (b, y + h / 2), color="#2a78d6")

    # ---------------- release guard + public output ---------------- #
    box(ax, 71.2, y, 13.6, h, "⑥ Release guard",
        "release.py\ncolumn whitelist\nshuffle · new IDs\nk-anonymity check", "release")
    arrow(ax, (xs[4] + w, y + h / 2), (71.2, y + h / 2), color="#eb6834")
    box(ax, 89, y, 10, h, "Public release",
        "mdav_k*.csv\n+ metadata.json\n(aggregated\ncolumns only)", "release", body_size=8)
    arrow(ax, (84.8, y + h / 2), (89, y + h / 2), color="#eb6834", lw=2.2)
    ax.text(78, 31.4, "✗ export blocked if not\nk-anonymous (e.g. IR)",
            fontsize=8, color="#c2410c", ha="center", va="top")

    # ---------------- evaluation ---------------- #
    ey, eh = 17, 9.5
    box(ax, 23, ey, 14.5, eh, "utility.py",
        "information loss (SSE/SST)\nmeans · variance · Gini\nregression · tax-rule checks", "eval")
    box(ax, 40, ey, 14.5, eh, "privacy.py",
        "k-anonymity · record linkage\ninterval disclosure · l-div /\nt-closeness · collusion attack",
        "eval")
    box(ax, 57, ey, 13.5, eh, "safe_display.py",
        "income bands\nsafe summaries only\n(used by app.py)", "eval")
    # AnonymizationResult "bus": the pipeline output feeds every evaluation module
    bus_y = 29.4
    arrow(ax, (xs[4] + w / 2, y), (xs[4] + w / 2, bus_y), color="#1baf7a", style="-", ls=(0, (4, 3)))
    ax.plot([30.25, xs[4] + w / 2], [bus_y, bus_y], color="#1baf7a", lw=1.6, ls=(0, (4, 3)), zorder=1)
    for x in (30.25, 47.25, xs[4] + w / 2):
        arrow(ax, (x, bus_y), (x, ey + eh), color="#1baf7a", ls=(0, (4, 3)))
    ax.text(41, bus_y + 0.45, "AnonymizationResult (original + anonymised, row-aligned)",
            fontsize=7.8, color="#127a55", ha="center", va="bottom", style="italic")

    # ---------------- experiments & reporting ---------------- #
    box(ax, 3, 17, 16.5, eh, "experiments.py",
        "k ∈ {3, 5, 10, 20} × 3 methods\nablation · scalability\n(O(n²d/k) check)", "report")
    arrow(ax, (23, ey + eh / 2), (19.5, ey + eh / 2), color="#4a3aa7")
    box(ax, 3, 5.2, 16.5, 8.6, "plotting.py",
        "9 figures\n→ results/figures/", "report")
    box(ax, 23, 5.2, 21, 8.6, "report_builder.py",
        "report_template.md + results\n→ docs/REPORT.md", "report")
    box(ax, 48, 5.2, 18, 8.6, "export_report.py",
        "REPORT.md → Word\n→ PDF (cover, TOC)", "report")
    arrow(ax, (11.25, 17), (11.25, 13.8), color="#4a3aa7")
    arrow(ax, (19.5, 9.5), (23, 9.5), color="#4a3aa7")
    arrow(ax, (44, 9.5), (48, 9.5), color="#4a3aa7")

    # ---------------- cross-cutting config ---------------- #
    box(ax, 69, 5.2, 16, 8.6, "config.py",
        "attribute roles · valid ranges\nAnonymizationConfig\n(k, method, seed)", "config",
        body_size=7.9)

    # ---------------- legend ---------------- #
    items = [("pipeline", "Anonymisation pipeline"), ("release", "Release / public output"),
             ("eval", "Evaluation & safe display"), ("report", "Experiments & reporting"),
             ("entry", "Entry points / config")]
    lx = 88.2
    ax.text(lx, 26.5, "Legend", fontsize=9.5, fontweight="bold", color=INK, va="center")
    for i, (kind, label) in enumerate(items):
        fill, edge = STYLES[kind]
        yy = 23.8 - i * 2.6
        ax.add_patch(FancyBboxPatch((lx, yy - 0.7), 1.8, 1.4, boxstyle="round,pad=0,rounding_size=0.3",
                                    facecolor=fill, edgecolor=edge, linewidth=1.3))
        ax.text(lx + 2.5, yy, label, fontsize=8, color=INK_2, va="center")
    ax.plot([lx, lx + 1.8], [10.0, 10.0], ls=(0, (6, 4)), color="#9a9994", lw=1.4)
    ax.text(lx + 2.5, 10.0, "Custodian\nboundary", fontsize=8, color=INK_2, va="center")

    out = ROOT / "docs"
    fig.savefig(out / "architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out / "architecture.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out / 'architecture.png'} and {out / 'architecture.svg'}")


if __name__ == "__main__":
    main()
