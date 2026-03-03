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


def make_year_dfs(agg: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return {year: DataFrame} — one tidy DataFrame per year.

    Each year DataFrame has columns:
      brand, building, [units], [capacity_kw], 1월, 2월, …, 12월, 연간합계

    Monthly values are kWh usage (already computed from cumulative diffs).
    Rows are sorted by 연간합계 descending.
    January correctly uses December of the prior year as its baseline because
    _compute_unit_usage processes all years in chronological sequence.
    """
    meta = ["brand", "building"]
    for c in ["units", "capacity_kw"]:
        if c in agg.columns:
            meta.append(c)

    year_dfs: dict[int, pd.DataFrame] = {}
    for y in EHP_YEARS:
        avail = [m for m in _MONTHS if f"usage_{y}_{m:02d}" in agg.columns]
        if not avail:
            continue
        src_cols = [f"usage_{y}_{m:02d}" for m in avail]
        ydf = agg[meta + src_cols].copy()
        rename = {f"usage_{y}_{m:02d}": f"{m}월" for m in avail}
        ydf = ydf.rename(columns=rename)
        mon_cols = [f"{m}월" for m in avail]
        ydf["연간합계"] = ydf[mon_cols].sum(axis=1, min_count=1).round(0)
        year_dfs[y] = ydf.sort_values("연간합계", ascending=False).reset_index(drop=True)
    return year_dfs


def _annual_totals(year_dfs: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Build brand × year annual-totals table from year_dfs."""
    if not year_dfs:
        return pd.DataFrame()
    # Union of all brands across all years
    all_brands = pd.concat(
        [ydf[["brand", "building"]] for ydf in year_dfs.values()]
    ).drop_duplicates().reset_index(drop=True)

    # Optionally carry over capacity_kw / units from whichever year has it
    for c in ["units", "capacity_kw"]:
        for ydf in year_dfs.values():
            if c in ydf.columns:
                all_brands = all_brands.merge(
                    ydf[["brand", "building", c]], on=["brand", "building"], how="left"
                )
                break

    for y, ydf in year_dfs.items():
        all_brands = all_brands.merge(
            ydf[["brand", "building", "연간합계"]].rename(columns={"연간합계": str(y)}),
            on=["brand", "building"],
            how="left",
        )

    year_cols = [str(y) for y in year_dfs]
    all_brands["Total"] = all_brands[year_cols].sum(axis=1, min_count=1).round(0)
    return all_brands.sort_values("Total", ascending=False).reset_index(drop=True)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _left_margin(df: pd.DataFrame) -> int:
    mx = df["brand"].astype(str).str.len().max() if len(df) else 20
    return min(max(int(mx) * 7, 120), 320)


def _hbar_layout(title: str, n_brands: int, left: int, **extra) -> dict:
    base = dict(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color="#222222"), x=0),
        height=max(380, n_brands * 22 + 120),
        xaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        yaxis=dict(automargin=True, zeroline=False,
                   tickfont=dict(size=10, color="#555555")),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom",
                    font=dict(size=11, color="#333333")),
        margin=dict(l=left, r=20, t=70, b=40),
    )
    base.update(extra)
    return base


# ─── Tab renderers ────────────────────────────────────────────────────────────

def _annual_tab(annual: pd.DataFrame) -> None:
    """Overall trend + per-brand stacked bar across all years."""
    year_cols = [c for c in annual.columns if c.isdigit()]
    if not year_cols:
        st.warning("No annual data available.")
        return

    # ── Year totals trend (all brands combined) ──
    year_totals = [(y, annual[y].sum(min_count=1)) for y in year_cols]
    xt, yt = zip(*year_totals)
    pct_changes = [None] + [
        round((yt[i] - yt[i-1]) / yt[i-1] * 100, 1) if yt[i-1] else None
        for i in range(1, len(yt))
    ]
    text_labels = [
        f"{v:,.0f}<br>({'+' if p > 0 else ''}{p}%)" if p is not None else f"{v:,.0f}"
        for v, p in zip(yt, pct_changes)
    ]
    fig_trend = go.Figure(go.Scatter(
        x=list(xt), y=list(yt),
        mode="lines+markers+text",
        text=text_labels, textposition="top center",
        line=dict(color="#4C72B0", width=2.5),
        marker=dict(size=9, color="#4C72B0"),
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig_trend.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>Total Annual Usage — All Brands</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=280,
        xaxis=dict(showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=11, color="#555555"),
                   dtick=1),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        margin=dict(l=70, r=20, t=55, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ── Per-brand stacked bar ──
    _n = len(annual)
    if _n >= 2:
        top_n = st.slider("Show top N brands", 5, _n, min(20, _n), key="ehp_annual_n")
    else:
        top_n = _n

    plot_df = annual.head(top_n).iloc[::-1].copy()
    fig = go.Figure()
    for i, year in enumerate(year_cols):
        fig.add_trace(go.Bar(
            y=plot_df["brand"], x=plot_df[year].fillna(0),
            name=year, orientation="h",
            marker_color=_PALETTE[i % len(_PALETTE)],
            marker_line_color="white", marker_line_width=0.5,
            hovertemplate=f"<b>%{{y}}</b><br>{year}: %{{x:,.0f}} kWh<extra></extra>",
        ))
    fig.update_layout(**_hbar_layout(
        f"Annual Usage by Brand — Top {top_n}", top_n, _left_margin(plot_df),
    ))
    st.plotly_chart(fig, use_container_width=True)

    show_cols = ["brand", "building"] + [c for c in ["units", "capacity_kw"] if c in annual.columns]
    show_cols += year_cols + ["Total"]
    out = add_display_index(annual[[c for c in show_cols if c in annual.columns]].copy())
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, "ehp_annual_usage.xlsx", "annual")


def _year_tab(year_dfs: dict) -> None:
    """Monthly breakdown for a selected year."""
    if not year_dfs:
        st.warning("No year data.")
        return

    available_years = sorted(year_dfs.keys())
    c1, c2 = st.columns([2, 3])
    with c1:
        sel_year = st.selectbox("Year", available_years,
                                index=len(available_years) - 1, key="ehp_year")
    year_df  = year_dfs[sel_year]
    mon_cols = [c for c in year_df.columns if c.endswith("월")]
    if not mon_cols:
        st.warning("No monthly data for selected year.")
        return

    _n = len(year_df)
    with c2:
        top_n = st.slider("Show top N brands", 5, _n, min(20, _n), key="ehp_year_n") if _n >= 2 else _n

    # ── Monthly totals (all brands) ──
    month_totals = year_df[mon_cols].sum()
    peak_m = month_totals.idxmax()
    low_m  = month_totals.idxmin()
    fig_mt = go.Figure(go.Bar(
        x=mon_cols, y=month_totals.values,
        marker_color=[
            "#C44E52" if c == peak_m else ("#55A868" if c == low_m else "#4C72B0")
            for c in mon_cols
        ],
        text=[f"{v:,.0f}" for v in month_totals.values],
        textposition="outside",
        textfont=dict(size=9, color="#666666"),
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig_mt.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>{sel_year} — Monthly Totals (All Brands)</b>"
                        f"   <span style='font-size:11px;color:#888'>peak={peak_m}, low={low_m}</span>",
                   font=dict(size=13, color="#222222"), x=0),
        height=300,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=11, color="#555555")),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        margin=dict(l=60, r=20, t=55, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig_mt, use_container_width=True)

    # ── Per-brand stacked bar ──
    plot_df = year_df.head(top_n).iloc[::-1].copy()
    fig = go.Figure()
    for i, col in enumerate(mon_cols):
        fig.add_trace(go.Bar(
            y=plot_df["brand"], x=plot_df[col].fillna(0),
            name=col, orientation="h",
            marker_color=_PALETTE[i % len(_PALETTE)],
            marker_line_color="white", marker_line_width=0.3,
            hovertemplate=f"<b>%{{y}}</b><br>{col}: %{{x:,.0f}} kWh<extra></extra>",
        ))
    fig.update_layout(**_hbar_layout(
        f"{sel_year} Monthly Usage by Brand — Top {top_n}", top_n, _left_margin(plot_df),
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ── Stats summary row ──
    stats = pd.DataFrame([{
        "Total kWh": f"{year_df['연간합계'].sum():,.0f}",
        "Peak month": peak_m,
        "Peak kWh": f"{month_totals[peak_m]:,.0f}",
        "Low month": low_m,
        "Low kWh": f"{month_totals[low_m]:,.0f}",
        "Avg / month": f"{month_totals.mean():,.0f}",
        "# brands": len(year_df),
    }])
    st.dataframe(stats, hide_index=True, use_container_width=True)

    out = add_display_index(year_df.copy())
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, f"ehp_year_{sel_year}.xlsx", "year_detail")


def _yoy_tab(annual: pd.DataFrame) -> None:
    """Year-over-year comparison: heatmap + change table."""
    year_cols = [c for c in annual.columns if c.isdigit()]
    if len(year_cols) < 2:
        st.warning("Need at least 2 years of data for YoY analysis.")
        return

    mode = st.radio(
        "View", ["Annual Usage (kWh)", "YoY Change (kWh)", "YoY Change (%)"],
        horizontal=True, key="ehp_yoy_mode",
    )

    brands = annual["brand"].tolist()

    if mode == "Annual Usage (kWh)":
        z_cols   = year_cols
        z_matrix = annual[year_cols].fillna(0).values.tolist()
        colorscale, zmid = "Blues", None
        fmt = ".0f"
    else:
        pairs   = [(year_cols[i-1], year_cols[i]) for i in range(1, len(year_cols))]
        z_cols  = [f"{a}→{b}" for a, b in pairs]
        rows = []
        for _, row in annual.iterrows():
            r = []
            for a, b in pairs:
                prev, curr = row.get(a, np.nan), row.get(b, np.nan)
                if mode == "YoY Change (kWh)":
                    r.append(round(curr - prev, 0) if pd.notna(curr) and pd.notna(prev) else np.nan)
                else:
                    r.append(round((curr - prev) / prev * 100, 1)
                             if pd.notna(curr) and pd.notna(prev) and prev != 0 else np.nan)
            rows.append(r)
        z_matrix   = rows
        colorscale = "RdBu_r"   # red = increase, blue = decrease
        zmid       = 0
        fmt        = ".1f" if "%" in mode else ".0f"

    hm_kwargs = dict(zmid=zmid) if zmid is not None else {}
    fig_hm = go.Figure(go.Heatmap(
        z=z_matrix,
        x=z_cols,
        y=brands,
        colorscale=colorscale,
        text=[[f"{v:{fmt}}" if pd.notna(v) else "" for v in row] for row in z_matrix],
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,.1f}<extra></extra>",
        **hm_kwargs,
    ))
    max_label = annual["brand"].astype(str).str.len().max() if len(annual) else 20
    left_margin = min(max(int(max_label) * 7, 120), 320)
    fig_hm.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>Brand × Year — {mode}</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=max(400, len(brands) * 18 + 120),
        xaxis=dict(tickfont=dict(size=10, color="#555555"), side="bottom"),
        yaxis=dict(automargin=True, tickfont=dict(size=9, color="#555555")),
        margin=dict(l=left_margin, r=80, t=55, b=60),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # ── YoY change table ──
    if mode != "Annual Usage (kWh)":
        tbl = annual[["brand", "building"]].copy()
        pairs = [(year_cols[i-1], year_cols[i]) for i in range(1, len(year_cols))]
        for a, b in pairs:
            diff = annual[b] - annual[a]
            pct  = (diff / annual[a] * 100).round(1)
            tbl[f"Δ {a}→{b} kWh"] = diff.round(0)
            tbl[f"Δ {a}→{b} %"]   = pct
        tbl["Total"] = annual["Total"]
        out = add_display_index(tbl)
        st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                     height=min(35 * len(out) + 38, 700))
        download_df_as_excel(out, "ehp_yoy_change.xlsx", "yoy")
    else:
        show = ["brand", "building"] + year_cols + ["Total"]
        out  = add_display_index(annual[[c for c in show if c in annual.columns]].copy())
        st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                     height=min(35 * len(out) + 38, 700))
        download_df_as_excel(out, "ehp_annual_usage.xlsx", "annual")


def _profile_tab(year_dfs: dict) -> None:
    """Seasonality: month (1-12) × year, total kWh per month across all brands."""
    if not year_dfs:
        st.warning("No data.")
        return

    # Compute (year, month) → total kWh aggregated across all brands
    records: list[dict] = []
    for year, ydf in sorted(year_dfs.items()):
        mon_cols = [c for c in ydf.columns if c.endswith("월")]
        for col in mon_cols:
            m = int(col.replace("월", ""))
            records.append({"year": year, "month": m, "month_label": col,
                            "total_kwh": ydf[col].sum(min_count=1)})
    profile = pd.DataFrame(records)
    if profile.empty:
        st.warning("No monthly data.")
        return

    # ── Line chart: month on x, one line per year ──
    fig = go.Figure()
    years = sorted(profile["year"].unique())
    for i, year in enumerate(years):
        yp = profile[profile["year"] == year].sort_values("month")
        fig.add_trace(go.Scatter(
            x=yp["month_label"], y=yp["total_kwh"],
            mode="lines+markers",
            name=str(year),
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2),
            marker=dict(size=6),
            hovertemplate=f"<b>{year}</b> %{{x}}: %{{y:,.0f}} kWh<extra></extra>",
        ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>Monthly Seasonality — All Brands Combined</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=420,
        xaxis=dict(title="Month", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=11, color="#555555")),
        yaxis=dict(title="kWh", showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        legend=dict(orientation="v", x=1.01, xanchor="left", y=1,
                    font=dict(size=11, color="#333333"),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA", borderwidth=1),
        margin=dict(l=70, r=120, t=55, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Average across years per month ──
    avg_by_month = (
        profile.groupby("month_label")["total_kwh"]
        .agg(["mean", "min", "max", "std"])
        .reindex([f"{m}월" for m in range(1, 13)]).dropna()
        .rename(columns={"mean": "Avg kWh", "min": "Min", "max": "Max", "std": "Std"})
    )
    avg_by_month["Peak year"] = profile.groupby("month_label").apply(
        lambda g: str(int(g.loc[g["total_kwh"].idxmax(), "year"]))
    ).reindex(avg_by_month.index)
    avg_by_month = avg_by_month.round(0).reset_index().rename(columns={"month_label": "Month"})

    st.markdown("**Monthly averages across all years**")
    st.dataframe(avg_by_month, hide_index=True, use_container_width=True)

    # ── Pivot table: year × month ──
    pivot = profile.pivot(index="year", columns="month_label", values="total_kwh")
    pivot = pivot.reindex(columns=[f"{m}월" for m in range(1, 13)], fill_value=np.nan)
    pivot["연간합계"] = pivot.sum(axis=1, min_count=1).round(0)
    pivot = pivot.reset_index().rename(columns={"year": "Year"})
    st.markdown("**Year × month pivot (kWh)**")
    st.dataframe(st_safe(pivot), hide_index=True, use_container_width=True)
    download_df_as_excel(pivot, "ehp_monthly_profile.xlsx", "profile")


# ─── Public entry point ───────────────────────────────────────────────────────

def render_ehp_view(df: pd.DataFrame) -> None:
    st.subheader("EHP(OAC) 전기 사용량 분석")
    st.caption("단위: kWh — 월별 누계 검침에서 월 사용량 산출 (January 2018 excluded — no prior baseline)")

    # ── Diagnostic: always-visible when something is wrong ──
    cum_cols      = [c for c in df.columns if c.startswith("cum_")]
    all_buildings = sorted(df["building"].dropna().unique().tolist())

    if df.empty or not cum_cols or not all_buildings:
        st.error("Parser returned no usable data. See details below.")
        st.write(f"**DataFrame shape:** {df.shape}")
        st.write(f"**Columns:** {list(df.columns)}")
        st.write(f"**Cumulative columns found:** {cum_cols}")
        if "building" in df.columns:
            st.write(f"**All building values (raw):** {df['building'].unique().tolist()}")
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10))
        return

    with st.expander("Debug — raw parsed data", expanded=False):
        st.write(f"Shape: {df.shape} | Cumulative cols: {len(cum_cols)} | Buildings: {all_buildings}")
        st.dataframe(df.head(5))

    # Building filter
    sel_bldg = st.multiselect(
        "Building", ["All"] + all_buildings, default=["All"], key="ehp_building",
    )
    active_bldg = all_buildings if "All" in sel_bldg else sel_bldg
    df = df[df["building"].isin(active_bldg)].copy()

    if df.empty:
        st.warning("No data for selected building.")
        return

    df_unit  = _compute_unit_usage(df)
    agg      = _agg_brand(df_unit)
    year_dfs = make_year_dfs(agg)       # {year: tidy DataFrame per year}
    annual   = _annual_totals(year_dfs) # brand × year annual totals

    tab_annual, tab_year, tab_yoy, tab_profile = st.tabs(
        ["Annual Summary", "Year Detail", "Year-over-Year", "Monthly Profile"]
    )

    with tab_annual:
        _annual_tab(annual)
    with tab_year:
        _year_tab(year_dfs)
    with tab_yoy:
        _yoy_tab(annual)
    with tab_profile:
        _profile_tab(year_dfs)
