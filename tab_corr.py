"""tab_corr.py — Correlation tab renderer for the 검침 내역 view."""
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats


_SUFFIX_ORDER = [
    "change", "pct",
    "current", "previous",
    "usage_m3", "usage_kw", "usage_m3_mwh",
    "usage_per_m2", "usage_per_py",
]
_SUFFIX_LABELS = {
    "previous":      "Previous Usage",
    "current":       "Current Usage",
    "usage_m3":      "Usage (m³)",
    "usage_kw":      "Usage (kWh)",
    "usage_m3_mwh":  "Usage (m³/MWh)",
    "usage_per_m2":  "Usage per m²",
    "usage_per_py":  "Usage per 평",
    "change":        "Quantitative Change",
    "pct":           "Percentage Change",
}
_COL_LABELS = {"size_m2": "m²", "size_py": "평 (py)"}
_UTIL_PREFIXES = ["water_", "hwater_", "elect_", "heat_"]  # trailing _ for startswith matching
from utils import BLD_COLOR as _BLDG_COLOR_MAP


def _col_label(col: str) -> str:
    if col in _COL_LABELS:
        return _COL_LABELS[col]
    for prefix in _UTIL_PREFIXES:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            return _SUFFIX_LABELS.get(suffix, suffix)
    return col


def _col_sort_key(col: str) -> int:
    for prefix in _UTIL_PREFIXES:
        if col.startswith(prefix):
            suffix = col[len(prefix):]
            try:
                return _SUFFIX_ORDER.index(suffix)
            except ValueError:
                return len(_SUFFIX_ORDER)
    return -1


def _col_util(col: str) -> str:
    for p in _UTIL_PREFIXES:
        if col.startswith(p):
            return p
    return col


def _build_cat_cols(all_numeric: list[str]) -> dict[str, list[str]]:
    cat_cols: dict[str, list[str]] = {}
    for cat, match in [
        ("m²",         lambda c: c == "size_m2"),
        ("평 (py)",     lambda c: c == "size_py"),
        ("Water",       lambda c: c.startswith("water_")),
        ("Hot Water",   lambda c: c.startswith("hwater_")),
        ("Electricity", lambda c: c.startswith("elect_")),
        ("Heat",        lambda c: c.startswith("heat_")),
    ]:
        cols = sorted([c for c in all_numeric if match(c)], key=_col_sort_key)
        if cols:
            cat_cols[cat] = cols
    return cat_cols


def _find_cat(col: str, cat_cols: dict) -> str | None:
    for cat, cols in cat_cols.items():
        if col in cols:
            return cat
    return None


def _render_auto_discovery(cur_df: pd.DataFrame, all_numeric: list[str], cat_cols: dict) -> None:
    st.subheader("Auto-Discover Correlations")

    scan_suffixes = {"change", "pct", "current"}
    scan_cols = [
        c for c in all_numeric
        if any(c.endswith(f"_{s}") for s in scan_suffixes)
        or c in ("size_m2", "size_py")
    ]

    disc_rows = []
    for i, ca in enumerate(scan_cols):
        for cb in scan_cols[i + 1:]:
            if _col_util(ca) == _col_util(cb):
                continue
            dp = cur_df[[ca, cb]].dropna()
            if len(dp) < 5:
                continue
            r, p = stats.pearsonr(dp[ca].values, dp[cb].values)
            disc_rows.append({
                "X":         ca,
                "Y":         cb,
                "r":         round(r, 3),
                "R²":        round(r ** 2, 3),
                "p-value":   round(p, 4),
                "n":         len(dp),
                "Direction": "positive" if r > 0 else "negative",
                "Strength":  (
                    "Strong"   if abs(r) >= 0.6 else
                    "Moderate" if abs(r) >= 0.35 else
                    "Weak"
                ),
            })

    if not disc_rows:
        st.info("Not enough cross-category data to run discovery.")
        return

    disc_df = pd.DataFrame(disc_rows).sort_values("R²", ascending=False).reset_index(drop=True)

    dc1, dc2, dc3 = st.columns([2, 2, 3])
    with dc1:
        min_r2 = st.slider("Min R²", 0.0, 1.0, 0.05, 0.05, key="disc_min_r2",
                           help="Only show pairs where R² is at least this value")
    with dc2:
        show_nonsig = st.checkbox("Include p ≥ 0.05", value=False, key="disc_show_nonsig",
                                  help="Also show pairs that are not statistically significant")
    with dc3:
        strength_filter = st.multiselect("Strength filter", ["Strong", "Moderate", "Weak"],
                                         default=["Strong", "Moderate"], key="disc_strength")

    shown = disc_df[
        (disc_df["R²"] >= min_r2) &
        (disc_df["Strength"].isin(strength_filter)) &
        (show_nonsig | (disc_df["p-value"] < 0.05))
    ].reset_index(drop=True)

    if shown.empty:
        st.info("No pairs match the current filters.")
    else:
        st.caption(f"{len(shown)} pair(s) found · Select a row to load it into the scatter below")
        disc_event = st.dataframe(
            shown,
            hide_index=True,
            use_container_width=True,
            on_select="rerun",
            selection_mode="single-row",
            column_config={
                "r":       st.column_config.NumberColumn("r",        format="%.3f"),
                "R²":      st.column_config.NumberColumn("R²",       format="%.3f"),
                "p-value": st.column_config.NumberColumn("p",        format="%.4f"),
                "X":       st.column_config.TextColumn("X Column",   width="medium"),
                "Y":       st.column_config.TextColumn("Y Column",   width="medium"),
            },
        )

        sel_rows = (
            disc_event.selection.rows
            if disc_event and hasattr(disc_event, "selection")
            else []
        )
        if sel_rows:
            sel = shown.iloc[sel_rows[0]]
            xc, yc = sel["X"], sel["Y"]
            xct, yct = _find_cat(xc, cat_cols), _find_cat(yc, cat_cols)
            if xct and yct:
                st.session_state["corr_x_cat"]      = xct
                st.session_state[f"corr_x_{xct}"]   = xc
                st.session_state["corr_y_cat"]       = yct
                st.session_state[f"corr_y_{yct}"]   = yc
                st.success(
                    f"Loaded **{_col_label(xc)}** vs **{_col_label(yc)}** "
                    f"(R² = {sel['R²']:.3f}) — scroll down to scatter"
                )

    # Correlation heatmap
    hm_cols = [c for c in scan_cols if c in cur_df.columns
               and (c.endswith("_change") or c.endswith("_pct"))]
    if len(hm_cols) >= 3:
        hm_data = cur_df[hm_cols].dropna()
        if len(hm_data) >= 3:
            corr_mat = hm_data.corr()
            labels = [_col_label(c) for c in corr_mat.columns]
            fig_hm = px.imshow(
                corr_mat,
                x=labels, y=labels,
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                text_auto=".2f",
                title="Correlation Matrix — Change & % Change",
                aspect="auto",
            )
            fig_hm.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                coloraxis_colorbar=dict(title="r"),
                font=dict(size=11),
            )
            fig_hm.update_traces(textfont_size=10)
            st.plotly_chart(fig_hm, use_container_width=True)


def _render_manual_scatter(cur_df: pd.DataFrame, cat_cols: dict, categories: list[str]) -> None:
    st.subheader("Manual Scatter")

    xc1, xc2, yc1, yc2, cc3 = st.columns(5)
    with xc1:
        x_cat = st.selectbox("X Category", categories, index=0, key="corr_x_cat")
    with xc2:
        x_col = st.selectbox("X Column", cat_cols[x_cat], index=0,
                             key=f"corr_x_{x_cat}", format_func=_col_label)
    with yc1:
        y_cat = st.selectbox("Y Category", categories, index=min(1, len(categories) - 1),
                             key="corr_y_cat")
    with yc2:
        y_col = st.selectbox("Y Column", cat_cols[y_cat], index=0,
                             key=f"corr_y_{y_cat}", format_func=_col_label)
    with cc3:
        color_by = st.selectbox("Color by", ["brand", "building"], index=0, key="corr_color")

    lc1, lc2, oc1, oc2 = st.columns([1, 1, 1, 4])
    with lc1:
        log_x = st.checkbox("Log X", value=False, key="corr_log_x")
    with lc2:
        log_y = st.checkbox("Log Y", value=False, key="corr_log_y")
    with oc1:
        remove_outliers = st.checkbox("Remove outliers", value=False, key="corr_remove_outliers")
    with oc2:
        iqr_k = st.slider("IQR multiplier", 0.5, 3.0, 1.5, 0.1,
                          key="corr_iqr_k", disabled=not remove_outliers)

    if x_col == y_col:
        st.info("Please select different columns for X and Y axes.")
        return

    hover_extra = [c for c in ["brand", "building", "size_m2", "size_py"]
                   if c in cur_df.columns and c not in [x_col, y_col]]
    corr_df = cur_df[[x_col, y_col] + hover_extra].dropna(subset=[x_col, y_col]).copy()

    if remove_outliers:
        for col in [x_col, y_col]:
            q1, q3 = corr_df[col].quantile(0.25), corr_df[col].quantile(0.75)
            iqr = q3 - q1
            corr_df = corr_df[(corr_df[col] >= q1 - iqr_k * iqr) & (corr_df[col] <= q3 + iqr_k * iqr)]

    if corr_df.empty:
        st.warning("No data with valid values for both selected columns.")
        return

    x_vals = corr_df[x_col].values.astype(float)
    y_vals = corr_df[y_col].values.astype(float)
    if log_x:
        mask = x_vals > 0
        x_vals, y_vals = np.log10(x_vals[mask]), y_vals[mask]
    if log_y:
        mask = y_vals > 0
        x_vals, y_vals = x_vals[mask], np.log10(y_vals[mask])

    plot_df = corr_df.copy()
    color_map = None
    if color_by == "building" and "building" in plot_df.columns:
        color_map = {b: _BLDG_COLOR_MAP.get(b, "#aaaaaa")
                     for b in plot_df["building"].astype(str).unique()}
    category_orders = (
        {color_by: sorted(plot_df[color_by].astype(str).unique())}
        if color_by in plot_df.columns else None
    )

    fig = px.scatter(
        plot_df,
        x=x_col, y=y_col,
        color=color_by if color_by in corr_df.columns else None,
        hover_data=hover_extra,
        log_x=log_x, log_y=log_y,
        title=f"{x_col} vs {y_col}",
        color_discrete_map=color_map,
        category_orders=category_orders,
    )

    reg_row = None
    if len(x_vals) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)

        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        y_line = slope * x_line + intercept
        if log_x:
            x_line = 10 ** x_line
        if log_y:
            y_line = 10 ** y_line

        fig.add_scatter(x=x_line, y=y_line, mode="lines", name="Trendline",
                        line=dict(color="red", width=2, dash="dash"))

        x_label = f"log10({x_col})" if log_x else x_col
        y_label = f"log10({y_col})" if log_y else y_col
        sign = "+" if intercept >= 0 else "-"
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            text=f"y = {slope:.4f}x {sign} {abs(intercept):.4f}",
            showarrow=False, align="left",
            bgcolor="rgba(255,255,255,0.8)", bordercolor="red",
            borderwidth=1, font=dict(size=12, color="red"),
        )
        reg_row = pd.DataFrame([{
            "equation":  f"{y_label} = {slope:.4f} × {x_label} + {intercept:.4f}",
            "slope":     round(slope, 6),
            "intercept": round(intercept, 6),
            "R²":        round(r_value ** 2, 6),
            "p-value":   f"{p_value:.4e}",
            "std_err":   round(std_err, 6),
            "n":         len(x_vals),
        }])

    fig.update_layout(height=550)
    st.plotly_chart(fig, use_container_width=True)

    if reg_row is not None:
        st.dataframe(reg_row, hide_index=True, width="stretch")

        r2 = r_value ** 2
        direction_meaning = (
            f"As **{x_col}** increases, **{y_col}** tends to **increase**."
            if slope > 0 else
            f"As **{x_col}** increases, **{y_col}** tends to **decrease**."
        )
        strength = (
            "very strong" if r2 >= 0.7 else
            "strong"      if r2 >= 0.5 else
            "moderate"    if r2 >= 0.3 else
            "weak"        if r2 >= 0.1 else
            "very weak"
        )
        sig_text = (
            "highly statistically significant (p < 0.001)" if p_value < 0.001 else
            "very statistically significant (p < 0.01)"    if p_value < 0.01  else
            "statistically significant (p < 0.05)"         if p_value < 0.05  else
            "**not statistically significant** (p ≥ 0.05) — treat this result with caution"
        )
        direction = "positive" if slope > 0 else "negative"
        st.markdown(f"""
**Interpretation**

There is a **{strength} {direction} linear relationship** between {x_col} and {y_col} (R² = {r2:.4f}). {direction_meaning}

The model explains **{r2*100:.1f}%** of the variance in {y_col}. The relationship is {sig_text}.

For every 1-unit increase in {x_col}, {y_col} changes by **{slope:.4f}** on average.
""")


def render_corr_tab(cur_df: pd.DataFrame) -> None:
    """Render the full Correlation tab content."""
    all_numeric = sorted([
        c for c in cur_df.columns
        if pd.api.types.is_numeric_dtype(cur_df[c]) and cur_df[c].notna().any()
    ])
    cat_cols = _build_cat_cols(all_numeric)
    categories = list(cat_cols.keys())

    if sum(len(v) for v in cat_cols.values()) < 2:
        st.info("Not enough numeric columns for correlation.")
        return

    _render_auto_discovery(cur_df, all_numeric, cat_cols)
    st.divider()
    _render_manual_scatter(cur_df, cat_cols, categories)
