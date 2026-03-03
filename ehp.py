import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import st_safe
from features import add_display_index, download_df_as_excel

EHP_YEARS   = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
_MONTHS     = list(range(1, 13))
_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

_PALETTE = [
    "#4C72B0", "#DD8A00", "#C44E52", "#55A868",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]

_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", color="#333333"),
)


# ─── Computation helpers ──────────────────────────────────────────────────────

def _compute_unit_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Add usage_YYYY_MM columns for each unit row (diff of consecutive cumulative readings).
    January 2018 has no prior baseline → NaN.
    Negative diffs (meter reset) are clipped to 0.
    """
    df = df.copy()
    prev_col = None
    for y in EHP_YEARS:
        for m in _MONTHS:
            cum_col   = f"cum_{y}_{m:02d}"
            usage_col = f"usage_{y}_{m:02d}"
            if cum_col in df.columns:
                if prev_col and prev_col in df.columns:
                    df[usage_col] = (df[cum_col] - df[prev_col]).clip(lower=0)
                else:
                    df[usage_col] = np.nan
                prev_col = cum_col
    return df


def _agg_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Sum per-unit usage columns → one row per brand+building."""
    usage_cols = [c for c in df.columns if c.startswith("usage_")]
    group_cols = ["brand", "building"]

    agg = df.groupby(group_cols, as_index=False)[usage_cols].sum(min_count=1)

    if "capacity_kw" in df.columns:
        cap = df.groupby(group_cols, as_index=False)["capacity_kw"].sum(min_count=1)
        agg = agg.merge(cap, on=group_cols, how="left")

    n_units = df.groupby(group_cols).size().reset_index(name="units")
    agg = agg.merge(n_units, on=group_cols, how="left")
    return agg


def _annual_totals(agg: pd.DataFrame) -> pd.DataFrame:
    """Sum monthly usages within each year → annual kWh per brand."""
    meta = ["brand", "building"]
    for c in ["units", "capacity_kw"]:
        if c in agg.columns:
            meta.append(c)
    result = agg[meta].copy()
    for y in EHP_YEARS:
        cols = [f"usage_{y}_{m:02d}" for m in _MONTHS if f"usage_{y}_{m:02d}" in agg.columns]
        if cols:
            result[str(y)] = agg[cols].sum(axis=1, min_count=1).round(0)
    return result


# ─── Tab renderers ────────────────────────────────────────────────────────────

def _annual_tab(annual: pd.DataFrame) -> None:
    year_cols = [c for c in annual.columns if c.isdigit()]
    if not year_cols:
        st.warning("No annual data available.")
        return

    annual = annual.copy()
    annual["Total"] = annual[year_cols].sum(axis=1, min_count=1).round(0)
    annual_sorted = annual.sort_values("Total", ascending=False).copy()

    _n = len(annual_sorted)
    if _n >= 2:
        top_n = st.slider("Show top N brands", 5, _n, min(20, _n), key="ehp_annual_n")
    else:
        top_n = _n

    plot_df = annual_sorted.head(top_n).iloc[::-1].copy()  # reversed: top at top

    fig = go.Figure()
    for i, year in enumerate(year_cols):
        fig.add_trace(go.Bar(
            y=plot_df["brand"],
            x=plot_df[year].fillna(0),
            name=year,
            orientation="h",
            marker_color=_PALETTE[i % len(_PALETTE)],
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=f"<b>%{{y}}</b><br>{year}: %{{x:,.0f}} kWh<extra></extra>",
        ))

    max_label = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(int(max_label) * 7, 120), 320)

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text=f"<b>Annual EHP Usage — Top {top_n}</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=max(420, top_n * 22 + 120),
        xaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        yaxis=dict(automargin=True, zeroline=False,
                   tickfont=dict(size=10, color="#555555")),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom",
                    font=dict(size=11, color="#333333")),
        margin=dict(l=left_margin, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table (all brands)
    show_cols = ["brand", "building"]
    for c in ["units", "capacity_kw"]:
        if c in annual_sorted.columns:
            show_cols.append(c)
    show_cols += year_cols + ["Total"]
    show = [c for c in show_cols if c in annual_sorted.columns]
    out = add_display_index(annual_sorted[show].copy())
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, "ehp_annual_usage.xlsx", "annual")


def _trend_tab(agg: pd.DataFrame) -> None:
    usage_cols = [c for c in agg.columns if c.startswith("usage_")]
    if not usage_cols:
        st.warning("No usage data.")
        return

    # Build sorted brand list
    agg = agg.copy()
    agg["_total"] = agg[usage_cols].sum(axis=1, min_count=1)
    sorted_brands = (
        agg.groupby("brand")["_total"].sum()
        .sort_values(ascending=False).index.tolist()
    )

    sel_brands = st.multiselect(
        "Select brands to compare", sorted_brands,
        default=sorted_brands[:min(5, len(sorted_brands))],
        key="ehp_trend_brands",
    )
    if not sel_brands:
        st.info("Select at least one brand.")
        return

    # All ordered periods that exist
    all_periods = [
        (y, m, f"usage_{y}_{m:02d}", f"{y}-{m:02d}")
        for y in EHP_YEARS for m in _MONTHS
        if f"usage_{y}_{m:02d}" in agg.columns
    ]

    fig = go.Figure()
    for i, brand in enumerate(sel_brands):
        rows = agg[agg["brand"] == brand]
        # Sum across buildings so one line per brand
        y_vals = [
            rows[col].sum(min_count=1) if col in rows.columns else np.nan
            for _, _, col, _ in all_periods
        ]
        x_vals = [period for _, _, _, period in all_periods]

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines+markers",
            name=brand,
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=4),
            hovertemplate=f"<b>{brand}</b><br>%{{x}}: %{{y:,.0f}} kWh<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>Monthly EHP Electricity Usage (kWh)</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=460,
        xaxis=dict(title="Period", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=9, color="#555555"), tickangle=-45),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1,
                    font=dict(size=10, color="#333333"),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA", borderwidth=1),
        margin=dict(l=60, r=160, t=55, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Downloadable table: brand × period
    tbl_rows = []
    for brand in sel_brands:
        rows = agg[agg["brand"] == brand]
        row = {"brand": brand}
        for _, _, col, period in all_periods:
            row[period] = round(rows[col].sum(min_count=1), 0) if col in rows.columns else np.nan
        tbl_rows.append(row)
    tbl = pd.DataFrame(tbl_rows)
    download_df_as_excel(tbl, "ehp_monthly_trend.xlsx", "trend")


def _year_tab(agg: pd.DataFrame) -> None:
    available_years = [
        y for y in EHP_YEARS
        if any(f"usage_{y}_{m:02d}" in agg.columns for m in _MONTHS)
    ]
    if not available_years:
        st.warning("No year data.")
        return

    c1, c2 = st.columns([2, 3])
    with c1:
        sel_year = st.selectbox("Year", available_years,
                                index=len(available_years) - 1, key="ehp_year")

    month_cols = [f"usage_{sel_year}_{m:02d}" for m in _MONTHS
                  if f"usage_{sel_year}_{m:02d}" in agg.columns]
    if not month_cols:
        st.warning("No data for selected year.")
        return

    _n = len(agg)
    with c2:
        if _n >= 2:
            top_n = st.slider("Show top N brands", 5, _n, min(20, _n), key="ehp_year_n")
        else:
            top_n = _n

    year_df = agg[["brand", "building"] + month_cols].copy()
    year_df["year_total"] = year_df[month_cols].sum(axis=1, min_count=1)
    year_df = year_df.sort_values("year_total", ascending=False).head(top_n)
    plot_df  = year_df.iloc[::-1].copy()  # reversed for horizontal bar

    fig = go.Figure()
    for i, (col, m) in enumerate(zip(month_cols, _MONTHS)):
        month_name = _MONTH_ABBR[m - 1]
        fig.add_trace(go.Bar(
            y=plot_df["brand"],
            x=plot_df[col].fillna(0),
            name=month_name,
            orientation="h",
            marker_color=_PALETTE[i % len(_PALETTE)],
            marker_line_color="white",
            marker_line_width=0.3,
            hovertemplate=f"<b>%{{y}}</b><br>{month_name}: %{{x:,.0f}} kWh<extra></extra>",
        ))

    max_label = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(int(max_label) * 7, 120), 320)

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text=f"<b>{sel_year} Monthly Usage — Top {top_n}</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=max(420, top_n * 22 + 120),
        xaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        yaxis=dict(automargin=True, zeroline=False,
                   tickfont=dict(size=10, color="#555555")),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom",
                    font=dict(size=11, color="#333333")),
        margin=dict(l=left_margin, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly totals row (all selected brands combined)
    totals = {_MONTH_ABBR[m - 1]: round(year_df[col].sum(), 0)
              for col, m in zip(month_cols, _MONTHS)}
    totals["Total"] = round(year_df["year_total"].sum(), 0)
    st.markdown(f"**Monthly totals — {sel_year} (shown brands)**")
    st.dataframe(pd.DataFrame([totals]), hide_index=True, use_container_width=True)

    # Per-brand table
    show = ["brand", "building"] + month_cols + ["year_total"]
    show = [c for c in show if c in year_df.columns]
    out = add_display_index(year_df.sort_values("year_total", ascending=False)[show].copy())
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, f"ehp_year_{sel_year}.xlsx", "year_detail")


# ─── Public entry point ───────────────────────────────────────────────────────

def render_ehp_view(df: pd.DataFrame) -> None:
    st.subheader("EHP(OAC) 전기 사용량 분석")
    st.caption("단위: kWh — 월별 누계 검침에서 월 사용량 산출 (January 2018 excluded — no prior baseline)")

    # Debug expander — shows raw parsed output before any filtering
    with st.expander("Debug — raw parsed data", expanded=False):
        st.write(f"Shape: {df.shape}")
        cum_cols = [c for c in df.columns if c.startswith("cum_")]
        st.write(f"Cumulative columns detected ({len(cum_cols)}):", cum_cols)
        st.write("Unique buildings:", df["building"].unique().tolist() if "building" in df.columns else "N/A")
        st.dataframe(df.head(5))

    # Building filter
    all_buildings = sorted(df["building"].dropna().unique().tolist())
    sel_bldg = st.multiselect(
        "Building", ["All"] + all_buildings, default=["All"], key="ehp_building",
    )
    active_bldg = all_buildings if "All" in sel_bldg else sel_bldg
    df = df[df["building"].isin(active_bldg)].copy()

    if df.empty:
        st.warning("No data for selected building.")
        with st.expander("Debug — raw parsed DataFrame"):
            st.write("Shape:", df.shape)
            st.write("Columns:", list(df.columns))
            st.dataframe(df.head(10))
        return

    df_unit = _compute_unit_usage(df)
    agg     = _agg_brand(df_unit)
    annual  = _annual_totals(agg)

    tab_annual, tab_trend, tab_year = st.tabs(
        ["Annual Summary", "Monthly Trend", "Year Detail"]
    )

    with tab_annual:
        _annual_tab(annual)
    with tab_trend:
        _trend_tab(agg)
    with tab_year:
        _year_tab(agg)
