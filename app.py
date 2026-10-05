"""
Streamlit dashboard - Micro-aggregation in Income and Tax Data Publication.

    streamlit run app.py

Safety rules enforced by the UI
-------------------------------
* Original confidential values are never displayed exactly unless the user
  switches on "Custodian view" (off by default, with a warning). Otherwise they
  are shown as bands / safe summaries (see microagg/safe_display.py).
* No per-record error is shown (released value + error = original value).
* Downloads go exclusively through release.build_release(), i.e. the release
  guard: identifiers dropped, rows shuffled, k-anonymity verified. A release that
  fails a blocking check cannot be downloaded.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from microagg import experiments, privacy, release, safe_display, utility
from microagg.anonymizer import anonymize
from microagg.config import (
    CATEGORICAL_QUASI_IDENTIFIERS, DERIVED_SENSITIVE, DIRECT_IDENTIFIERS, EXPERIMENT_K_VALUES,
    METHOD_LABELS, METHODS, MONETARY_ATTRIBUTES, NUMERIC_QUASI_IDENTIFIERS, RANDOM_SEED,
    SENSITIVE_NUMERIC, AnonymizationConfig,
)
from microagg.data_generator import generate_income_tax_data
from microagg.preprocessing import DataValidationError, validate_and_clean

st.set_page_config(page_title="Micro-aggregation · Income & Tax Data", page_icon="🔒", layout="wide")

METHOD_COLORS = {"mdav": "#2a78d6", "individual_ranking": "#eb6834", "random": "#1baf7a"}
ORIGINAL_GREY = "#a3a29c"
ANON_BLUE = "#2a78d6"


# =============================================================================
# Cached computation
# =============================================================================
@st.cache_data(show_spinner="Generating synthetic taxpayers…")
def load_synthetic(n: int, seed: int) -> pd.DataFrame:
    return generate_income_tax_data(n=n, seed=seed)


@st.cache_data(show_spinner=False)
def load_upload(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content))


@st.cache_data(show_spinner="Validating and cleaning…")
def clean_data(raw: pd.DataFrame, numeric: tuple, categorical: tuple, ids: tuple):
    return validate_and_clean(raw, numeric, categorical, direct_identifiers=ids)


def make_config(params: dict) -> AnonymizationConfig:
    return AnonymizationConfig(**params)


@st.cache_data(show_spinner="Micro-aggregating…", max_entries=24)
def run_pipeline(clean: pd.DataFrame, params: dict):
    res = anonymize(clean, make_config(params))
    return res, utility.compute_utility(res), privacy.compute_privacy(res)


@st.cache_data(show_spinner="Running the k × method grid…", max_entries=8)
def run_grid(clean: pd.DataFrame, params: dict, k_values: tuple, methods: tuple) -> pd.DataFrame:
    summary, _ = experiments.run_grid(clean, k_values=k_values, methods=methods, base=make_config(params))
    return summary


# =============================================================================
# Small helpers
# =============================================================================
def pct(v, digits=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}%"


def num(v, digits=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{digits}f}"


def chart(c: alt.Chart):
    """Render with Streamlit's theme (axis/text colours follow light or dark mode)."""
    st.altair_chart(c.configure_view(strokeWidth=0).configure_legend(orient="bottom"), width="stretch")


def explain(text: str):
    st.caption(text)


# =============================================================================
# Sidebar - inputs
# =============================================================================
st.sidebar.title("🔒 Micro-aggregation")
st.sidebar.caption("Income & tax microdata publication")

source = st.sidebar.radio("Data source", ["Synthetic dataset", "Upload CSV"], horizontal=True)
if source == "Synthetic dataset":
    n_records = st.sidebar.slider("Number of taxpayers", 500, 10_000, 5000, step=500)
    seed = int(st.sidebar.number_input("Random seed", 0, 10_000, RANDOM_SEED))
    raw = load_synthetic(n_records, seed)
else:
    seed = int(st.sidebar.number_input("Random seed", 0, 10_000, RANDOM_SEED))
    up = st.sidebar.file_uploader("CSV file", type="csv",
                                  help="Expected columns: see README › Dataset. Other schemas work "
                                       "if you select the attributes below.")
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch to the synthetic dataset.")
        st.stop()
    try:
        raw = load_upload(up.getvalue())
    except Exception as exc:  # malformed CSV
        st.error(f"Could not read the CSV: {exc}")
        st.stop()

cols = list(raw.columns)
num_default = [c for c in NUMERIC_QUASI_IDENTIFIERS + SENSITIVE_NUMERIC if c in cols]
cat_default = [c for c in CATEGORICAL_QUASI_IDENTIFIERS if c in cols]
with st.sidebar.expander("Attributes to micro-aggregate (= released)", expanded=False):
    numeric_attrs = st.multiselect("Numeric", cols, default=num_default)
    categorical_attrs = st.multiselect("Categorical", [c for c in cols if c not in numeric_attrs],
                                       default=[c for c in cat_default if c not in numeric_attrs])
    id_attrs = st.multiselect("Direct identifiers (always dropped)", cols,
                              default=[c for c in DIRECT_IDENTIFIERS if c in cols])
    st.caption("Columns not selected as numeric/categorical are **never** released.")

st.sidebar.subheader("Anonymisation")
method = st.sidebar.selectbox("Method", METHODS, format_func=METHOD_LABELS.get)
k = st.sidebar.slider("k (minimum group size)", 2, 50, 5)
with st.sidebar.expander("Distance space (advanced)"):
    cat_weight = st.slider("Categorical weight w", 0.0, 2.0, float(1 / np.sqrt(2)), 0.05, format="%.3f",
                           help="Weight of one-hot columns. w = 0.707 makes a category mismatch "
                                "cost the same as a 1-σ numeric difference.")
    log_monetary = st.checkbox("log1p() monetary attributes in the distance", value=True,
                               help="Grouping only - released values are always raw-scale means.")
st.sidebar.divider()
custodian = st.sidebar.toggle("Custodian view: show exact original values", value=False)
if custodian:
    st.sidebar.warning("Exact original values are now visible on screen. Never screenshot or "
                       "share this view - it is for the data custodian only.")

# =============================================================================
# Validation
# =============================================================================
if not numeric_attrs and not categorical_attrs:
    st.error("Select at least one attribute to micro-aggregate.")
    st.stop()
try:
    clean, vreport = clean_data(raw, tuple(numeric_attrs), tuple(categorical_attrs), tuple(id_attrs))
except DataValidationError as exc:
    st.error(f"Input validation failed: {exc}")
    st.stop()

params = dict(k=int(k), method=method, numeric_attributes=tuple(numeric_attrs),
              categorical_attributes=tuple(categorical_attrs), categorical_weight=float(cat_weight),
              log_transform_monetary=bool(log_monetary), seed=seed)
try:
    make_config(params).validate(len(clean))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

result, U, P = run_pipeline(clean, params)
cfg = result.config

# =============================================================================
# Header
# =============================================================================
st.title("Micro-aggregation in Income and Tax Data Publication")
st.markdown(
    f"**{METHOD_LABELS[method]}** with **k = {k}** on **{len(clean):,}** records · "
    f"{len(cfg.numeric_attributes)} numeric + {len(cfg.categorical_attributes)} categorical attributes. "
    "Micro-aggregation *reduces* disclosure risk; it does **not** make data absolutely anonymous - "
    "see the *Privacy & attacks* tab."
)

tabs = st.tabs(["① Data & validation", "② Anonymise & compare", "③ Utility", "④ Privacy & attacks",
                "⑤ Privacy–utility trade-off", "⑥ Export", "ℹ️ About & limitations"])

# =============================================================================
# ① Data & validation
# =============================================================================
with tabs[0]:
    s = vreport.summary()
    c = st.columns(5)
    c[0].metric("Input rows", f"{s['input_rows']:,}")
    c[1].metric("Clean records", f"{s['output_rows']:,}")
    c[2].metric("Duplicates removed", s["duplicate_rows_removed"] + s["duplicate_ids_removed"])
    c[3].metric("Cells imputed", s["cells_imputed"])
    c[4].metric("Cells repaired (identity)", s["cells_repaired_by_identity"])
    for w in vreport.warnings:
        st.warning(w)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Validation report")
        explain("Counts per column. Invalid entries (non-numeric, out of range, rule violations) are set "
                "to missing, then repaired via taxable = gross − deductions where exact, otherwise imputed "
                "(numeric: median within occupation; categorical: mode).")
        st.dataframe(vreport.to_frame(), hide_index=True)
    with right:
        st.subheader("Attribute roles")
        roles = []
        for col in cols:
            if col in id_attrs:
                r = "Direct identifier — dropped"
            elif col in DERIVED_SENSITIVE:
                r = "Derived sensitive — not released"
            elif col in numeric_attrs or col in categorical_attrs:
                r = ("Sensitive — micro-aggregated" if col in SENSITIVE_NUMERIC
                     else "Quasi-identifier — micro-aggregated")
            else:
                r = "Not selected — not released"
            roles.append({"column": col, "role": r})
        st.dataframe(pd.DataFrame(roles), hide_index=True)

    st.subheader("Original data — safe summary")
    explain("No minimum/maximum is shown (each would be one person's exact value); figures are rounded "
            "to 3 significant digits.")
    if cfg.numeric_attributes:
        st.dataframe(safe_display.safe_summary(clean, cfg.numeric_attributes), hide_index=True)
    if custodian:
        st.subheader("Raw input (custodian view)")
        st.dataframe(raw.head(50))
    else:
        st.info("Record-level original data is hidden. Enable *Custodian view* in the sidebar to inspect it.")

# =============================================================================
# ② Anonymise & compare
# =============================================================================
with tabs[1]:
    c = st.columns(5)
    if result.is_joint:
        sizes = result.group_sizes
        c[0].metric("Groups", f"{result.n_groups:,}")
        c[1].metric("Group size min / max", f"{sizes.min()} / {sizes.max()}")
    else:
        c[0].metric("Groups", "per attribute")
        c[1].metric("Group size min", f"{k} (per attribute)")
    c[2].metric("Runtime", f"{result.runtime_seconds:.2f} s")
    c[3].metric("k-anonymity level", int(P["k_anonymity_level"]),
                delta="satisfied" if P["k_anonymity_satisfied"] else "NOT satisfied",
                delta_color="normal" if P["k_anonymity_satisfied"] else "inverse")
    c[4].metric("Information loss (raw)", pct(U["il_percent"]))

    st.subheader("Original vs released records")
    explain("Random sample. Original monetary values are shown as bands and ages as decades "
            "(custodian view shows exact values). Per-record errors are deliberately not shown: "
            "released value + error would reveal the original. Every released row is the centroid "
            "of its group, so all members of a group show the same released values.")
    n_show = st.slider("Records to show", 5, 30, 10, key="n_show")
    sample_seed = st.number_input("Sample seed", 0, 9999, 0, key="sample_seed")
    st.dataframe(safe_display.comparison_table(result, n=n_show, seed=int(sample_seed),
                                               reveal_original=custodian))

    if result.is_joint:
        st.subheader("Group-size distribution")
        gs = pd.Series(result.group_sizes).value_counts().sort_index().rename_axis("size").reset_index(name="groups")
        chart(alt.Chart(gs).mark_bar(color=ANON_BLUE, cornerRadiusEnd=3, size=28).encode(
            x=alt.X("size:O", title="Group size", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("groups:Q", title="Number of groups"),
            tooltip=["size", "groups"]).properties(height=220))
        explain(f"MDAV guarantees k ≤ |G| ≤ 2k−1: all groups have exactly k = {k} members except possibly "
                "the last one.")

    st.subheader("Released dataset preview")
    rel, checks = release.build_release(result, raise_on_fail=False)
    if any(ch.blocking and not ch.passed for ch in checks):
        st.error("This configuration is **not releasable** (see the Export tab) — preview withheld.")
    else:
        st.dataframe(rel.head(25), hide_index=True)
        explain("Rows are shuffled and carry a fresh release_id; taxpayer_id and tax_bracket are not included.")

# =============================================================================
# ③ Utility
# =============================================================================
with tabs[2]:
    c = st.columns(5)
    c[0].metric("IL raw scale", pct(U["il_percent"]), help="100·SSE/SST on z-scored raw values.")
    c[1].metric("IL log scale", pct(U.get("il_log_percent")), help="Same, after log1p of monetary values.")
    c[2].metric("Categorical cells changed", pct(U.get("categorical_change_%")))
    c[3].metric("Correlation distortion", num(U.get("correlation_distortion")),
                help="Mean |r − r'| over attribute pairs.")
    c[4].metric("Regression CI overlap", num(U.get("regression_ci_overlap"), 2),
                help="1 = identical inference (Karr et al. 2006).")

    st.subheader("Per-attribute statistics")
    explain("Group means preserve every attribute's mean (and total) exactly; variance shrinks because "
            "within-group variation is removed.")
    st.dataframe(utility.attribute_table(result).style.format({
        "mean_original": "{:,.2f}", "mean_anonymized": "{:,.2f}", "mean_rel_diff_%": "{:.3f}",
        "variance_ratio": "{:.3f}", "ks_statistic": "{:.3f}", "median_abs_rel_error_%": "{:.2f}"}),
        hide_index=True)

    left, right = st.columns(2)
    with left:
        if "gross_income" in cfg.numeric_attributes:
            st.subheader("Gross-income distribution")
            o = result.original["gross_income"].astype(float)
            lo, hi = o.quantile(0.01), o.quantile(0.99)
            dens = pd.concat([
                pd.DataFrame({"log10_income": np.log10(o[(o >= lo) & (o <= hi)]), "data": "Original"}),
                pd.DataFrame({"log10_income": np.log10(result.anonymized["gross_income"].clip(lo, hi)),
                              "data": "Released"}),
            ])
            chart(alt.Chart(dens).transform_density(
                "log10_income", groupby=["data"], as_=["log10_income", "density"], bandwidth=0.05,
                extent=[float(np.log10(lo)), float(np.log10(hi))]).mark_line(strokeWidth=2).encode(
                x=alt.X("log10_income:Q", title="log10(gross income, ₹)"),
                y=alt.Y("density:Q", title="Density"),
                color=alt.Color("data:N", scale=alt.Scale(domain=["Original", "Released"],
                                                          range=[ORIGINAL_GREY, ANON_BLUE]), title=None),
                tooltip=["data:N", alt.Tooltip("log10_income:Q", format=".2f"),
                         alt.Tooltip("density:Q", format=".3f")]).properties(height=280))
            explain("Shown for the original P1–P99 range only, so outliers cannot be read off the axis.")
    with right:
        etr_o = utility.effective_tax_by_quintile(result.original)
        etr_a = utility.effective_tax_by_quintile(result.anonymized)
        if etr_o is not None:
            st.subheader("Effective tax rate by income quintile")
            etr = pd.DataFrame({"Original": etr_o, "Released": etr_a}).rename_axis("quintile").reset_index() \
                .melt("quintile", var_name="data", value_name="rate")
            chart(alt.Chart(etr).mark_bar(cornerRadiusEnd=3).encode(
                x=alt.X("quintile:N", title="Gross-income quintile (Q1 = poorest)",
                        axis=alt.Axis(labelAngle=0)),
                xOffset="data:N", y=alt.Y("rate:Q", title="Effective tax rate (%)"),
                color=alt.Color("data:N", scale=alt.Scale(domain=["Original", "Released"],
                                                          range=[ORIGINAL_GREY, ANON_BLUE]), title=None),
                tooltip=["quintile", "data", alt.Tooltip("rate:Q", format=".2f")]).properties(height=280))

    st.subheader("Income & tax statistics used by analysts")
    so, sa = utility.income_statistics(result.original), utility.income_statistics(result.anonymized)
    if so:
        st.dataframe(pd.DataFrame({"original": so, "released": sa}).style.format("{:,.3f}"))
    tc = utility.tax_consistency(result.anonymized)
    if tc:
        c = st.columns(2)
        c[0].metric("Records violating taxable = gross − deductions", pct(tc.get("identity_violation_%")))
        c[1].metric("Records whose tax ≠ schedule(released income)", pct(tc.get("tax_rule_violation_%")))
        explain("Averaging preserves linear identities (when one partition is used for every attribute) but "
                "not the non-linear tax schedule. Individual ranking breaks even the linear identity because "
                "each attribute is grouped separately.")

    num_cols = list(cfg.numeric_attributes)
    if len(num_cols) >= 2:
        st.subheader("Correlation change (released − original)")
        diff = (result.anonymized[num_cols].corr() - result.original[num_cols].corr()).fillna(0)
        d = diff.rename_axis("a").reset_index().melt("a", var_name="b", value_name="delta")
        lim = max(0.05, float(d["delta"].abs().max()))
        chart(alt.Chart(d).mark_rect(stroke="white", strokeWidth=2).encode(
            x=alt.X("a:N", title=None, sort=num_cols, axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("b:N", title=None, sort=num_cols),
            color=alt.Color("delta:Q", scale=alt.Scale(scheme="redblue", domain=[-lim, lim], reverse=True),
                            title="Δ r"),
            tooltip=["a", "b", alt.Tooltip("delta:Q", format="+.3f")]).properties(height=380))

    reg = utility.regression_validity(result)
    if reg:
        st.subheader("Regression validity: log(1 + gross income) ~ demographics")
        st.markdown(f"R² original **{reg['r2_original']:.3f}** · released **{reg['r2_anonymized']:.3f}** · "
                    f"mean 95% CI overlap **{reg['mean_ci_overlap']:.2f}**")
        st.dataframe(reg["table"].style.format("{:.4f}"))
        explain("R² usually *rises* after aggregation (noise is averaged away) — that is an artefact, "
                "not better data. Look at coefficient bias and CI overlap instead.")

# =============================================================================
# ④ Privacy & attacks
# =============================================================================
with tabs[3]:
    st.info("**Micro-aggregation is not absolute anonymity.** k-anonymity bounds the chance of picking out "
            "*which* record is yours; it does not stop an attacker from learning *something about you* "
            "(attribute disclosure) or from combining the release with what they already know.")
    c = st.columns(4)
    c[0].metric("Smallest equivalence class", int(P["k_anonymity_level"]),
                delta=f"required ≥ {k}", delta_color="off")
    c[1].metric("Linkage success — attacker knows all", pct(P["dbrl_all_%"]),
                help="Distance-based record linkage with the original records as external data.")
    c[2].metric("Linkage success — attacker knows QIs", pct(P["dbrl_qi_%"]))
    c[3].metric("Max. per-record risk", num(P["dbrl_all_max_record_risk"], 3), delta=f"1/k = {1 / k:.3f}",
                delta_color="off")

    st.subheader("Attribute disclosure")
    tol = st.slider("Tolerance for 'value disclosed' (±%)", 1, 30, 10) / 100
    inter = privacy.interval_disclosure(result, p=tol)
    if inter:
        c = st.columns(len(inter))
        for col_, (attr, v) in zip(c, inter.items()):
            col_.metric(f"{attr} within ±{tol:.0%}", pct(v))
        explain("Share of taxpayers whose released value lies within the tolerance of their true value. "
                "Because every record in a group carries the same values, an attacker only needs to know the "
                "*group* a person falls into — not the exact record — to learn this.")
    hom = privacy.homogeneity(result)
    if hom:
        c = st.columns(4)
        c[0].metric("Records in bracket-homogeneous groups", pct(hom["homogeneous_records_%"]))
        c[1].metric("Min / mean distinct l-diversity", f"{hom['min_l_diversity']} / {hom['mean_l_diversity']:.2f}")
        c[2].metric("Max t-closeness (bracket EMD)", num(hom["max_t_closeness"]))
        c[3].metric("Mean t-closeness", num(hom["mean_t_closeness"]))
        explain("Homogeneity attack: if all members of a group share the same tax bracket, identifying the "
                "group reveals the bracket with certainty, whatever k is.")
    dom = privacy.dominance_rules(result)
    if dom:
        c = st.columns(2)
        c[0].metric("Groups failing (1, 50%) dominance", pct(dom["dominance_groups_%"], 2))
        c[1].metric("Groups failing p%-rule (p = 10)", pct(dom["p_rule_groups_%"], 2))
        explain("When one very high earner dominates a group, the published mean ≈ that person's income ÷ |G|.")
    inf = privacy.attribute_inference_attack(result, tolerance=tol)
    if inf:
        c = st.columns(3)
        c[0].metric("QI-based income inference success", pct(inf["inference_success_%"]))
        c[1].metric("Baseline (no release, guess median)", pct(inf["baseline_success_%"]))
        c[2].metric("Disclosure gain", f"{inf['disclosure_gain_pp']:+.1f} pp")
        explain("The attacker knows a target's age, dependents, gender, marital status, education, occupation, "
                "region and city tier, links to the closest released record and reads off its income.")

    st.subheader("Background-knowledge (collusion) attack")
    coll = privacy.collusion_attack(result, fractions=(0.1, 0.25, 0.5, 0.75, 0.9, 0.95))
    if coll is not None:
        explain("The attacker already knows the true income of a share f of all taxpayers. When every other "
                "member of a target's group is known, the target's income follows exactly: "
                "x_t = |G|·mean(G) − Σ others. Success ≈ f^(|G|−1): it shrinks with k but never reaches zero.")
        long = coll.melt("known_fraction", value_vars=["exactly_recovered_%", "expected_%"],
                         var_name="series", value_name="pct")
        long["series"] = long["series"].map({"exactly_recovered_%": "Simulated", "expected_%": "Expected f^(|G|−1)"})
        chart(alt.Chart(long).mark_line(point=alt.OverlayMarkDef(size=60, filled=True)).encode(
            x=alt.X("known_fraction:Q", title="Share of population known to attacker", axis=alt.Axis(format="%")),
            y=alt.Y("pct:Q", title="Targets recovered exactly (%)"),
            color=alt.Color("series:N", scale=alt.Scale(domain=["Simulated", "Expected f^(|G|−1)"],
                                                        range=[ANON_BLUE, ORIGINAL_GREY]), title=None),
            strokeDash=alt.StrokeDash("series:N", legend=None),
            tooltip=[alt.Tooltip("known_fraction:Q", format=".0%"), "series", alt.Tooltip("pct:Q", format=".2f")],
        ).properties(height=280))

# =============================================================================
# ⑤ Trade-off
# =============================================================================
with tabs[4]:
    st.markdown("Runs every selected method for every selected k with the current attribute and distance "
                "settings, then plots utility loss against disclosure risk.")
    c1, c2 = st.columns(2)
    k_sel = c1.multiselect("k values", sorted(set(EXPERIMENT_K_VALUES) | {k, 2, 15, 30, 50}),
                           default=sorted(set(EXPERIMENT_K_VALUES) | {k}))
    m_sel = c2.multiselect("Methods", METHODS, default=list(METHODS), format_func=METHOD_LABELS.get)
    k_sel = [kk for kk in k_sel if kk <= len(clean)]
    grid_key = json.dumps([params, sorted(k_sel), m_sel, len(clean), seed], sort_keys=True, default=str)
    if st.button("Run comparison", type="primary"):
        st.session_state["grid_key"] = grid_key
    if st.session_state.get("grid_key") not in (None, grid_key):
        st.info("Settings changed since the last comparison — press *Run comparison* to refresh.")
    elif st.session_state.get("grid_key") == grid_key:
        if not k_sel or not m_sel:
            st.warning("Choose at least one k and one method.")
        else:
            summ = run_grid(clean, params, tuple(sorted(k_sel)), tuple(m_sel))
            summ["Method"] = summ["method"].map(METHOD_LABELS)
            color = alt.Color("Method:N", scale=alt.Scale(domain=[METHOD_LABELS[m] for m in METHODS],
                                                          range=[METHOD_COLORS[m] for m in METHODS]), title=None)
            sc = alt.Chart(summ).encode(
                x=alt.X("il_percent:Q", title="Information loss IL (%) → worse utility"),
                y=alt.Y("dbrl_all_%:Q", title="Record-linkage success (%) → worse privacy"),
                color=color, detail="Method:N",
                tooltip=["Method", "k", alt.Tooltip("il_percent:Q", format=".2f", title="IL %"),
                         alt.Tooltip("dbrl_all_%:Q", format=".2f", title="DBRL %"),
                         alt.Tooltip("interval_disclosure_gross_income_%:Q", format=".1f",
                                     title="income ±10% %")])
            chart((sc.mark_line(strokeWidth=2) + sc.mark_point(size=80, filled=True)
                   + sc.mark_text(dx=12, dy=-8, fontSize=11).encode(text="k:Q", color=alt.value("#8a8985")))
                  .properties(height=360, title="Privacy–utility trade-off (labels = k)"))
            cc = st.columns(2)
            for col_, (metric, title) in zip(cc, [("il_log_percent", "IL, log scale (%)"),
                                                   ("interval_disclosure_gross_income_%", "Income disclosed within ±10% (%)")]):
                with col_:
                    base = alt.Chart(summ).encode(x=alt.X("k:Q", title="k"), y=alt.Y(f"{metric}:Q", title=title),
                                                  color=color, tooltip=["Method", "k", alt.Tooltip(f"{metric}:Q", format=".2f")])
                    chart((base.mark_line(strokeWidth=2) + base.mark_point(size=60, filled=True)).properties(height=260))
            show = ["Method", "k", "il_percent", "il_log_percent", "categorical_change_%", "regression_ci_overlap",
                    "k_anonymity_level", "dbrl_all_%", "dbrl_qi_%", "interval_disclosure_gross_income_%",
                    "homogeneous_records_%", "inference_success_%", "collusion_f0.5_%"]
            st.dataframe(summ[[c for c in show if c in summ.columns]].style.format(precision=2), hide_index=True)

# =============================================================================
# ⑥ Export
# =============================================================================
with tabs[5]:
    st.subheader("Release guard")
    rel, checks = release.build_release(result, raise_on_fail=False)
    st.dataframe(pd.DataFrame([{"check": ch.name, "blocking": ch.blocking,
                                "status": "✅ pass" if ch.passed else "❌ FAIL", "detail": ch.detail}
                               for ch in checks]), hide_index=True)
    blocked = any(ch.blocking and not ch.passed for ch in checks)
    if blocked:
        st.error("Export disabled: the anonymised data failed a blocking check. Individual ranking aggregates "
                 "each attribute separately, so records stay unique on the combination of attributes.")
    else:
        st.success("All blocking checks passed. Only micro-aggregated attributes, in shuffled order, are exported.")
        metrics = {"config": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in params.items()},
                   "utility": U, "privacy": P}
        c = st.columns(3)
        c[0].download_button("⬇ Anonymised dataset (CSV)", rel.to_csv(index=False).encode("utf-8"),
                             file_name=f"income_tax_{method}_k{k}.csv", mime="text/csv", type="primary")
        c[1].download_button("⬇ Release notes (JSON)",
                             json.dumps(release.release_metadata(result), indent=2).encode("utf-8"),
                             file_name=f"income_tax_{method}_k{k}_metadata.json", mime="application/json")
        c[2].download_button("⬇ Metrics (JSON)", json.dumps(metrics, indent=2, default=float).encode("utf-8"),
                             file_name=f"metrics_{method}_k{k}.json", mime="application/json")
        explain("The metrics file contains aggregate statistics only (no record-level values).")

# =============================================================================
# About
# =============================================================================
with tabs[6]:
    st.markdown(r"""
### What this dashboard does
1. **Validates** the input (schema, types, ranges, edit rules), repairs values exactly where an accounting
   identity allows it, and imputes the rest.
2. **Partitions** the records into groups of at least *k* similar taxpayers with **MDAV**
   (or a baseline), in a standardised distance space (log-scaled money, one-hot categories).
3. **Aggregates**: every record is replaced by its group centroid (mean for numbers, mode for categories).
4. **Measures** utility and disclosure risk, and **guards** the export.

### MDAV in one paragraph
While at least 3k records remain: find the record *r* farthest from the centroid, and the record *s* farthest
from *r*; form a group from *r* and its k−1 nearest neighbours, then from *s* and its k−1 nearest neighbours.
Finish with one or two groups so every group has between k and 2k−1 records. Cost: **O(n²·d / k)** time,
**O(n·d)** memory.

### What it guarantees — and what it does not
* ✅ **k-anonymity on the released attributes** (MDAV / random partition): each released record is identical to
  at least k−1 others, so linkage picks the right person with probability ≤ 1/k.
* ✅ Means and totals (e.g. total tax revenue) are preserved exactly; linear identities survive.
* ❌ **Attribute disclosure**: groups of similar taxpayers have similar incomes — learning someone's group reveals
  their income approximately (homogeneity / similarity attack).
* ❌ **Background knowledge**: an attacker who knows the other k−1 members of a group recovers the remaining
  member's value exactly from the published mean.
* ❌ **Outliers**: a very high earner dominates their group mean.
* ❌ **Multiple releases**: publishing the same data with different k or attribute sets enables differencing.
* ❌ No formal guarantee against arbitrary auxiliary information (unlike differential privacy).

See `docs/ALGORITHM.md` and `docs/REPORT.md` for the full discussion.
""")
