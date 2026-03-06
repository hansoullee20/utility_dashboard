from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import st_safe
from features import add_display_index, download_df_as_excel, get_simple_floors, parse_floor_value
from viz import plot_hist_with_tails

# Color palette — mirrors viz.py
_WATER_COLOR = "#4C72B0"
_ELECT_COLOR = "#DD8A00"
_HEAT_COLOR  = "#C44E52"
_TOTAL_COLOR = "#7B68EE"   # medium slate blue for 총 합계
_GRID        = "#DDDDDD"

_UTIL_COLS = [
    ("water_total", "상하수도",  _WATER_COLOR),
    ("elect_total", "전기요금",  _ELECT_COLOR),
    ("heat_total",  "열요금",   _HEAT_COLOR),
]

_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial, sans-serif", color="#333333"),
)

# ─── Shared selector data ─────────────────────────────────────────────────────

_VIEW_SEGMENTS = {
    ("상하수도", "합계"):        [("water_excl",     "전용",       _WATER_COLOR),
                                  ("water_comm",     "공용",       "#89AAD4")],
    ("상하수도", "전용"):        [("water_excl",     "전용",       _WATER_COLOR)],
    ("상하수도", "공용"):        [("water_comm",     "공용",       "#89AAD4")],
    ("전기요금", "합계"):        [("elect_excl",     "전용",       _ELECT_COLOR),
                                  ("elect_comm",     "공용",       "#EDB96A")],
    ("전기요금", "전용"):        [("elect_excl",     "전용",       _ELECT_COLOR)],
    ("전기요금", "공용"):        [("elect_comm",     "공용",       "#EDB96A")],
    ("열요금",   "합계"):        [("hvac_excl",      "냉난방 전용", _HEAT_COLOR),
                                  ("hvac_comm",      "냉난방 공용", "#E08080"),
                                  ("hotwater_excl",  "급탕 전용",  "#8B3A3A"),
                                  ("hotwater_comm",  "급탕 공용",  "#C47C7C")],
    ("열요금",   "냉난방 합계"): [("hvac_excl",      "냉난방 전용", _HEAT_COLOR),
                                  ("hvac_comm",      "냉난방 공용", "#E08080")],
    ("열요금",   "냉난방 전용"): [("hvac_excl",      "냉난방 전용", _HEAT_COLOR)],
    ("열요금",   "냉난방 공용"): [("hvac_comm",      "냉난방 공용", "#E08080")],
    ("열요금",   "급탕 합계"):   [("hotwater_excl",  "급탕 전용",  "#8B3A3A"),
                                  ("hotwater_comm",  "급탕 공용",  "#C47C7C")],
    ("열요금",   "급탕 전용"):   [("hotwater_excl",  "급탕 전용",  "#8B3A3A")],
    ("열요금",   "급탕 공용"):   [("hotwater_comm",  "급탕 공용",  "#C47C7C")],
    ("총 합계",  "합계"):        [("total_excl",     "전용",       _TOTAL_COLOR),
                                  ("total_comm",     "공용",       "#B0A8F0")],
    ("총 합계",  "전용"):        [("total_excl",     "전용",       _TOTAL_COLOR)],
    ("총 합계",  "공용"):        [("total_comm",     "공용",       "#B0A8F0")],
}

_VIEW_OPTIONS = {
    "상하수도": ["합계", "전용", "공용"],
    "전기요금": ["합계", "전용", "공용"],
    "열요금":   ["합계", "냉난방 합계", "냉난방 전용", "냉난방 공용",
                 "급탕 합계", "급탕 전용", "급탕 공용"],
    "총 합계":  ["합계", "전용", "공용"],
}


_TABLE_EXTRA = {
    ("상하수도", "합계"):        ["water_total"],
    ("상하수도", "전용"):        ["water_comm",    "water_total"],
    ("상하수도", "공용"):        ["water_excl",    "water_total"],
    ("전기요금", "합계"):        ["elect_total"],
    ("전기요금", "전용"):        ["elect_comm",    "elect_total"],
    ("전기요금", "공용"):        ["elect_excl",    "elect_total"],
    ("열요금",   "합계"):        ["heat_total"],
    ("열요금",   "냉난방 합계"): ["heat_total"],
    ("열요금",   "냉난방 전용"): ["hvac_comm",     "heat_total"],
    ("열요금",   "냉난방 공용"): ["hvac_excl",     "heat_total"],
    ("열요금",   "급탕 합계"):   ["heat_total"],
    ("열요금",   "급탕 전용"):   ["hotwater_comm", "heat_total"],
    ("열요금",   "급탕 공용"):   ["hotwater_excl", "heat_total"],
    ("총 합계",  "합계"):        ["total"],
    ("총 합계",  "전용"):        ["total_comm",    "total"],
    ("총 합계",  "공용"):        ["total_excl",    "total"],
}


def _util_selector(df: pd.DataFrame, key: str):
    """Render utility + view selectors. Returns (sel_util, view_mode, segments)."""
    available = [k for k in _VIEW_OPTIONS
                 if any(c in df.columns for segs in
                        [_VIEW_SEGMENTS[(k, v)] for v in _VIEW_OPTIONS[k]]
                        for c, _, _ in segs)]
    if not available:
        return None, None, []

    sel_util = st.radio("Utility", available, horizontal=True, key=f"{key}_util")

    if sel_util == "열요금":
        r1, r2 = st.columns(2)
        with r1:
            heat_cat = st.radio("Category", ["합계", "냉난방", "급탕"],
                                horizontal=True, key=f"{key}_heat_cat")
        with r2:
            if heat_cat == "합계":
                st.empty()
                heat_sub = "합계"
            else:
                # Share the same key as the non-열요금 view radio so the
                # 전용/공용 level stays consistent when switching utilities.
                heat_sub = st.radio("Detail", ["합계", "전용", "공용"],
                                    horizontal=True, key=f"{key}_view")
        view_mode = heat_cat if heat_cat == "합계" else f"{heat_cat} {heat_sub}"
    else:
        view_mode = st.radio("View", _VIEW_OPTIONS[sel_util],
                             horizontal=True, key=f"{key}_view")

    segments = [(c, lbl, clr)
                for c, lbl, clr in _VIEW_SEGMENTS[(sel_util, view_mode)]
                if c in df.columns]
    return sel_util, view_mode, segments


# ─── Public entry point ───────────────────────────────────────────────────────

def render_billing_view(df: pd.DataFrame) -> None:
    st.subheader("수도광열비 부과 내역")
    st.caption("단위: 만원 (VAT 별도)")

    # ── Filters ──
    all_buildings = sorted(df["building"].dropna().unique().tolist())
    all_floors    = get_simple_floors(df)

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        sel_bldg = st.multiselect(
            "Building", ["All"] + all_buildings,
            default=["All"], key="billing_building",
        )
    with fc2:
        sel_floor = st.multiselect(
            "Floor", ["All"] + all_floors,
            default=["All"], key="billing_floor",
        )
    with fc3:
        has_gong = df["brand"].astype(str).str.contains("공실", na=False).any()
        gong_mode = st.radio(
            "공실 filter", ["All", "Exclude 공실", "공실 only"],
            horizontal=True, key="billing_gongshil",
            disabled=not has_gong,
        )

    active_bldg = all_buildings if "All" in sel_bldg else sel_bldg
    bldg_df = df[df["building"].isin(active_bldg)].copy()

    if "All" not in sel_floor and sel_floor and "floor" in bldg_df.columns:
        sel_set = set(sel_floor)
        mask = bldg_df["floor"].apply(
            lambda v: bool(set(parse_floor_value(str(v))) & sel_set)
        )
        bldg_df = bldg_df[mask].copy()

    if gong_mode == "공실 only":
        fdf = bldg_df[bldg_df["brand"].astype(str).str.contains("공실", na=False)].copy()
    elif gong_mode == "Exclude 공실":
        fdf = bldg_df[~bldg_df["brand"].astype(str).str.contains("공실", na=False)].copy()
    else:
        fdf = bldg_df.copy()

    if fdf.empty:
        st.warning("No data for the selected filters.")
        return

    # ── Report download ────────────────────────────────────────────────────────
    with st.expander("Download Report (PDF)", expanded=False):
        st.caption("현재 필터가 적용된 청구 데이터를 기반으로 업무용 PDF 보고서를 생성합니다.")
        lang_billing = st.radio("Language", ["한국어 (ko)", "English (en)"],
                                horizontal=True, key="billing_report_lang")
        if st.button("Generate PDF Report", key="billing_gen_report"):
            from billing_report import generate_billing_pdf
            with st.spinner("Generating PDF…"):
                pdf_bytes = generate_billing_pdf(
                    fdf,
                    context={"date": date.today(), "buildings": sel_bldg},
                    lang="ko" if lang_billing.startswith("한") else "en",
                )
            bldg_tag = "all" if "All" in sel_bldg else "_".join(sel_bldg)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"billing_report_{bldg_tag}_{date.today()}.pdf",
                mime="application/pdf",
                key="billing_dl_report",
            )

    # ── Tabs ──
    tab_rank, tab_hist, tab_bldg, tab_comp, tab_ratio, tab_perm2 = st.tabs([
        "Billing Ranking", "Histogram", "Building Summary",
        "Composition", "공용/전용 Ratio", "Per m²",
    ])

    with tab_rank:
        _ranking_tab(fdf)
    with tab_hist:
        _histogram_tab(bldg_df)  # passes unfiltered-by-공실 so tab can control it
    with tab_bldg:
        _building_tab(fdf)
    with tab_comp:
        _composition_tab(fdf)
    with tab_ratio:
        _ratio_tab(fdf)
    with tab_perm2:
        _per_m2_tab(fdf)


# ─── Tab renderers ────────────────────────────────────────────────────────────

def _ranking_tab(df: pd.DataFrame) -> None:
    st.subheader("Billing Ranking")

    _n = len(df)
    if _n >= 2:
        top_n = st.slider("Show top N brands in chart", min(10, _n - 1), _n, min(30, _n), key="rank_n")
    else:
        top_n = _n
    sel_util, view_mode, segments = _util_selector(df, key="rank")
    if not segments:
        st.warning("No utility cost columns found.")
        return

    seg_cols = [c for c, _, _ in segments]
    extra    = _TABLE_EXTRA.get((sel_util, view_mode), [])

    # x-axis anchored to the full-total column so all views (합계/전용/공용) share the same scale
    _ref_col = {"상하수도": "water_total", "전기요금": "elect_total", "열요금": "heat_total"}.get(sel_util)
    x_max = df[_ref_col].fillna(0).max() * 1.05 if _ref_col and _ref_col in df.columns else None

    sort_key = st.radio(
        "Sort by", ["현재 뷰 (Current view)", "합계 (Total)"],
        horizontal=True, key="rank_sort",
    )
    if sort_key == "합계 (Total)" and _ref_col and _ref_col in df.columns:
        sorted_df = df.sort_values(_ref_col, ascending=False).copy()
    else:
        sort_series = df[[c for c in seg_cols if c in df.columns]].fillna(0).sum(axis=1)
        sorted_df = df.assign(_sort=sort_series).sort_values("_sort", ascending=False).drop(columns="_sort").copy()

    # ── Chart (top N, reversed so highest is at top) ──
    plot_df = sorted_df.head(top_n).iloc[::-1].copy()

    fig = go.Figure()
    for col, label, color in segments:
        fig.add_trace(go.Bar(
            y=plot_df["brand"],
            x=plot_df[col].fillna(0),
            name=label,
            orientation="h",
            marker_color=color,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:,.0f}} 만원<extra></extra>",
        ))

    max_label_len = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(max_label_len * 7, 120), 320)
    title_view = "" if view_mode == "합계" else f" ({view_mode})"
    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text=f"<b>{sel_util}{title_view} — Top {top_n}</b>", font=dict(size=13, color="#222222"), x=0),
        height=max(420, top_n * 22 + 100),
        xaxis=dict(
            title="만원",
            showgrid=True, gridcolor=_GRID, griddash="dot",
            zeroline=False, tickfont=dict(size=10, color="#555555"),
            **( {"range": [0, x_max]} if x_max else {} ),
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=10, color="#555555"),
            automargin=True,
        ),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11, color="#333333")),
        margin=dict(l=left_margin, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Full ranked table (all brands, with context columns) ──
    context_cols = ["building", "floor", "unit", "size_m2"]
    # Any *합계 view: total first, then breakdown components
    # Sublevel views: selected col → total → other extra (opposite 전용/공용)
    if view_mode.endswith("합계"):
        tbl_cols = ["brand"] + extra + seg_cols + context_cols
    else:
        total_first = [c for c in extra if c.endswith("_total") or c == "total"]
        other_extra = [c for c in extra if c not in total_first]
        tbl_cols = ["brand"] + seg_cols + total_first + other_extra + context_cols
    show = list(dict.fromkeys(c for c in tbl_cols if c in sorted_df.columns))
    out = add_display_index(sorted_df[show].copy())

    label = view_mode if view_mode != "합계" else sel_util
    st.markdown(f"**{len(out)} brands** — sorted by {label} (high → low)")
    st.dataframe(
        st_safe(out), hide_index=True, use_container_width=True,
        height=min(35 * len(out) + 38, 700),
    )
    download_df_as_excel(out, filename=f"billing_ranking_{sel_util}_{view_mode}.xlsx", sheet_name="ranking")


# Reference total column per utility (used when 합계 is selected)
_HIST_REF_COL = {
    "상하수도": "water_total",
    "전기요금": "elect_total",
    "열요금":   "heat_total",
    "총 합계":  "total",
}


def _histogram_tab(df: pd.DataFrame) -> None:
    st.subheader("Histogram")

    bins     = st.session_state.get("bins", 50)
    tail_pct = st.session_state.get("tail", 20)

    # ── Local 공실 filter ──
    has_gong = df["brand"].astype(str).str.contains("공실", na=False).any()
    gong_mode = st.radio(
        "공실", ["All", "Exclude 공실", "공실 only"],
        horizontal=True, key="hist_gong", disabled=not has_gong,
    )
    if gong_mode == "공실 only":
        df = df[df["brand"].astype(str).str.contains("공실", na=False)].copy()
    elif gong_mode == "Exclude 공실":
        df = df[~df["brand"].astype(str).str.contains("공실", na=False)].copy()

    sel_util, view_mode, segments = _util_selector(df, key="hist")
    if not segments:
        st.warning("No cost columns found.")
        return

    # Resolve to a single column for histogramming
    if view_mode == "합계":
        ref = _HIST_REF_COL.get(sel_util)
        val_col = ref if ref and ref in df.columns else segments[0][0]
    else:
        val_col = segments[0][0]

    if val_col not in df.columns:
        st.info("No data for selected column.")
        return

    s = df[val_col].dropna()
    if s.empty:
        st.info("No data for selected column.")
        return

    lo = float(s.quantile(tail_pct / 100))
    hi = float(s.quantile(1 - tail_pct / 100))

    title = f"{sel_util} {view_mode} (만원)"
    display_cols = [c for c in ["brand", val_col, "building", "floor", "unit", "size_m2"] if c in df.columns]
    plot_hist_with_tails(
        s, bins=int(bins), lo=lo, hi=hi,
        title=title,
        source_df=df, val_col=val_col,
        key=f"billing_hist_{sel_util}_{view_mode}",
        display_cols=display_cols,
        tail_pct=tail_pct,
    )

    # ── Tail table ──
    show_mode = st.radio(
        "Show", ["All", "Top", "Middle", "Bottom"],
        horizontal=True, key="hist_show",
    )
    label = f"{tail_pct}%"

    if show_mode == "All":
        tbl = df[display_cols].dropna(subset=[val_col]).sort_values(val_col, ascending=False).copy()
        st.markdown(f"**All entries** — sorted high → low ({len(tbl)})")
    elif show_mode == "Top":
        tbl = df[df[val_col] >= hi][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**Top {label} (≥ {hi:,.2f})** — sorted high → low ({len(tbl)})")
    elif show_mode == "Middle":
        tbl = df[(df[val_col] > lo) & (df[val_col] < hi)][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**Middle** ({lo:,.2f} – {hi:,.2f}) — sorted high → low ({len(tbl)})")
    else:  # Bottom
        tbl = df[df[val_col] <= lo][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**Bottom {label} (≤ {lo:,.2f})** — sorted high → low ({len(tbl)})")

    tbl = add_display_index(tbl)
    st.dataframe(st_safe(tbl), hide_index=True, use_container_width=True,
                 height=min(35 * len(tbl) + 38, 700))
    download_df_as_excel(tbl, filename=f"billing_hist_{sel_util}_{view_mode}_{show_mode}.xlsx", sheet_name="hist")


def _building_tab(df: pd.DataFrame) -> None:
    st.subheader("Building Summary")

    present = [(c, lbl, clr) for c, lbl, clr in _UTIL_COLS if c in df.columns]
    sum_cols = [c for c, _, _ in present] + (["total"] if "total" in df.columns else [])

    if not sum_cols:
        st.warning("No cost columns found.")
        return

    agg = df.groupby("building")[sum_cols].sum().reset_index()
    sort_col = next((c for c in ["total"] + [c for c, _, _ in present] if c in agg.columns), None)
    if sort_col:
        agg = agg.sort_values(sort_col, ascending=False)

    # Stacked vertical bar
    fig = go.Figure()
    for col, label, color in present:
        if col not in agg.columns:
            continue
        fig.add_trace(go.Bar(
            x=agg["building"],
            y=agg[col],
            name=label,
            marker_color=color,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=f"<b>%{{x}}동</b><br>{label}: %{{y:,.0f}} 만원<extra></extra>",
        ))

    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text="<b>Utility Cost by Building</b>", font=dict(size=13, color="#222222"), x=0),
        height=400,
        xaxis=dict(title="Building", tickfont=dict(size=12, color="#555555")),
        yaxis=dict(
            title="만원",
            showgrid=True, gridcolor=_GRID, griddash="dot",
            zeroline=False, tickfont=dict(size=10, color="#555555"),
        ),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11, color="#333333")),
        margin=dict(l=10, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Pie — total bill share per building
    if "total" in agg.columns:
        building_colors = [_WATER_COLOR, _ELECT_COLOR, _HEAT_COLOR, "#7FA87F", "#9B59B6"]
        fig_pie = go.Figure(go.Pie(
            labels=agg["building"],
            values=agg["total"],
            hole=0.35,
            textinfo="label+percent",
            hovertemplate="<b>%{label}동</b><br>%{value:,.0f} 만원 (%{percent})<extra></extra>",
            marker_colors=building_colors[:len(agg)],
        ))
        fig_pie.update_layout(
            **_BASE_LAYOUT,
            title=dict(text="<b>Total Bill Share by Building</b>", font=dict(size=13, color="#222222"), x=0),
            height=360,
            margin=dict(l=10, r=10, t=55, b=10),
            showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("**Building totals**")
    st.dataframe(st_safe(agg), hide_index=True, use_container_width=True)
    download_df_as_excel(agg, filename="billing_building_summary.xlsx", sheet_name="building")


# ─── Composition tab ──────────────────────────────────────────────────────────

_COMP_COLS = [
    ("water_total", "상하수도", _WATER_COLOR),
    ("elect_total", "전기요금", _ELECT_COLOR),
    ("heat_total",  "열요금",   _HEAT_COLOR),
]


def _composition_tab(df: pd.DataFrame) -> None:
    """Show each brand's cost split by utility (absolute stacked bar + % table)."""
    st.subheader("Cost Composition by Brand")

    present = [(c, lbl, clr) for c, lbl, clr in _COMP_COLS if c in df.columns]
    if not present:
        st.warning("No utility cost columns found.")
        return

    seg_cols = [c for c, _, _ in present]

    mode = st.radio("Chart mode", ["100% stacked (share)", "Absolute (만원)"],
                    horizontal=True, key="comp_mode")

    _n = len(df)
    top_n = st.slider("Show top N brands", min(10, _n), _n, min(40, _n), key="comp_n") if _n > 1 else _n

    sort_col = next((c for c in ["total"] + seg_cols if c in df.columns), seg_cols[0])
    plot_df = df.sort_values(sort_col, ascending=False).head(top_n).iloc[::-1].copy()

    if mode.startswith("100%"):
        row_sums = plot_df[seg_cols].fillna(0).sum(axis=1).replace(0, float("nan"))
        for c in seg_cols:
            plot_df[f"_{c}_pct"] = (plot_df[c].fillna(0) / row_sums * 100).round(1)
        x_col   = lambda c: f"_{c}_pct"
        x_title = "% of total bill"
        x_range = [0, 100]
        htmpl   = lambda lbl: f"<b>%{{y}}</b><br>{lbl}: %{{x:.1f}}%<extra></extra>"
    else:
        x_col   = lambda c: c
        x_title = "만원"
        x_range = None
        htmpl   = lambda lbl: f"<b>%{{y}}</b><br>{lbl}: %{{x:,.0f}} 만원<extra></extra>"

    fig = go.Figure()
    for col, label, color in present:
        fig.add_trace(go.Bar(
            y=plot_df["brand"],
            x=plot_df[x_col(col)].fillna(0),
            name=label,
            orientation="h",
            marker_color=color,
            marker_line_color="white",
            marker_line_width=0.5,
            hovertemplate=htmpl(label),
        ))

    max_label_len = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(max_label_len * 7, 120), 320)
    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="stack",
        title=dict(text=f"<b>Utility Cost Composition — Top {top_n}</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=max(420, top_n * 22 + 100),
        xaxis=dict(
            title=x_title,
            showgrid=True, gridcolor=_GRID, griddash="dot",
            zeroline=False, tickfont=dict(size=10, color="#555555"),
            **( {"range": x_range} if x_range else {} ),
        ),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#555555"),
                   automargin=True),
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom",
                    font=dict(size=11, color="#333333")),
        margin=dict(l=left_margin, r=20, t=70, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Composition table
    tbl = df.sort_values(sort_col, ascending=False).copy()
    row_sums_all = tbl[seg_cols].fillna(0).sum(axis=1).replace(0, float("nan"))
    for col, lbl, _ in present:
        tbl[f"{col}_pct"] = (tbl[col].fillna(0) / row_sums_all * 100).round(1)

    pct_cols = [f"{c}_pct" for c, _, _ in present]
    show_cols = ["brand", "building", "floor"] + seg_cols + pct_cols + (["total"] if "total" in tbl.columns else [])
    show_cols = [c for c in show_cols if c in tbl.columns]
    out = add_display_index(tbl[show_cols].copy())
    st.markdown(f"**{len(out)} brands** — sorted by total cost (high → low)")
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, filename="billing_composition.xlsx", sheet_name="composition")


# ─── 공용/전용 Ratio tab ──────────────────────────────────────────────────────

_RATIO_PAIRS = [
    ("water_comm",    "water_excl",   "water_total",  "상하수도",  "#89AAD4", _WATER_COLOR),
    ("elect_comm",    "elect_excl",   "elect_total",  "전기요금",  "#EDB96A", _ELECT_COLOR),
    ("hvac_comm",     "hvac_excl",    "heat_total",   "냉난방",    "#E08080", _HEAT_COLOR),
    ("hotwater_comm", "hotwater_excl","heat_total",   "급탕",      "#C47C7C", "#8B3A3A"),
    ("total_comm",    "total_excl",   "total",        "총 합계",   "#B0A8F0", _TOTAL_COLOR),
]


def _ratio_tab(df: pd.DataFrame) -> None:
    """공용 vs 전용 ratio analysis — flag brands with disproportionate common charges."""
    st.subheader("공용 / 전용 Cost Ratio")
    st.caption("Brands where 공용 (common area) charges represent an unusually large share of their bill.")

    # Only show utilities where both comm and excl columns exist
    available = [(comm, excl, tot, lbl, c_comm, c_excl)
                 for comm, excl, tot, lbl, c_comm, c_excl in _RATIO_PAIRS
                 if comm in df.columns and excl in df.columns]
    if not available:
        st.warning("No 공용/전용 columns found.")
        return

    util_labels = [lbl for _, _, _, lbl, _, _ in available]
    sel = st.radio("Utility", util_labels, horizontal=True, key="ratio_util")
    comm, excl, tot_col, lbl, c_comm, c_excl = next(r for r in available if r[3] == sel)

    wdf = df.copy()
    denom = wdf[comm].fillna(0) + wdf[excl].fillna(0)
    wdf["comm_ratio"] = (wdf[comm].fillna(0) / denom.replace(0, float("nan")) * 100).round(1)
    wdf["excl_ratio"] = (100 - wdf["comm_ratio"]).round(1)

    # Outlier threshold: p75 + 1.5 * IQR (upper fence)
    ratios = wdf["comm_ratio"].dropna()
    q25, q75 = ratios.quantile(0.25), ratios.quantile(0.75)
    iqr = q75 - q25
    upper_fence = min(q75 + 1.5 * iqr, 100.0)
    median_ratio = ratios.median()

    _n = len(wdf)
    top_n = st.slider("Show top N brands", min(10, _n), _n, min(40, _n), key="ratio_n") if _n > 1 else _n
    sort_col_r = tot_col if tot_col in wdf.columns else comm
    plot_df = wdf.sort_values(sort_col_r, ascending=False).head(top_n).copy()
    plot_df = plot_df.sort_values("comm_ratio", ascending=True)  # sort by ratio for chart

    colors = ["#C44E52" if r >= upper_fence else c_comm
              for r in plot_df["comm_ratio"].fillna(0)]

    fig = go.Figure(go.Bar(
        y=plot_df["brand"],
        x=plot_df["comm_ratio"].fillna(0),
        orientation="h",
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=0.5,
        text=[f"{v:.1f}%" for v in plot_df["comm_ratio"].fillna(0)],
        textposition="outside",
        textfont=dict(size=9, color="#666666"),
        hovertemplate="<b>%{y}</b><br>공용 ratio: %{x:.1f}%<extra></extra>",
        name="공용 비율",
    ))

    fig.add_vline(x=float(median_ratio), line_dash="dot", line_color=_HEAT_COLOR,
                  line_width=1.5, annotation_text=f"Median {median_ratio:.1f}%",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=_HEAT_COLOR))
    fig.add_vline(x=float(upper_fence), line_dash="dash", line_color="#C44E52",
                  line_width=1.5, annotation_text=f"Outlier fence {upper_fence:.1f}%",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#C44E52"))

    max_label_len = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(max_label_len * 7, 120), 320)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>{lbl} — 공용 비율 (%) · red = outlier ≥ {upper_fence:.1f}%</b>",
                   font=dict(size=13, color="#222222"), x=0),
        height=max(420, top_n * 22 + 100),
        xaxis=dict(title="공용 비율 (%)", range=[0, 105],
                   showgrid=True, gridcolor=_GRID, griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#555555"),
                   automargin=True),
        showlegend=False,
        margin=dict(l=left_margin, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    n_outliers = int((wdf["comm_ratio"] >= upper_fence).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brands analyzed", len(wdf.dropna(subset=["comm_ratio"])))
    c2.metric("Median 공용 ratio", f"{median_ratio:.1f}%")
    c3.metric("Outlier fence", f"{upper_fence:.1f}%")
    c4.metric("Outlier brands", n_outliers)

    # Table: all brands with flag
    tbl = wdf.sort_values("comm_ratio", ascending=False).copy()
    tbl["outlier"] = tbl["comm_ratio"] >= upper_fence
    show_cols = ["brand", "building", "floor", comm, excl, "comm_ratio", "outlier"]
    if tot_col in tbl.columns:
        show_cols.insert(4, tot_col)
    show_cols = [c for c in show_cols if c in tbl.columns]
    out = add_display_index(tbl[show_cols].copy())
    st.markdown(f"**{n_outliers} outlier brand(s)** with 공용 ratio ≥ {upper_fence:.1f}%")
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, filename=f"billing_ratio_{lbl}.xlsx", sheet_name="ratio")


# ─── Per m² tab ───────────────────────────────────────────────────────────────

_PERM2_COLS = [
    ("total",       "총 합계",  _TOTAL_COLOR),
    ("water_total", "상하수도", _WATER_COLOR),
    ("elect_total", "전기요금", _ELECT_COLOR),
    ("heat_total",  "열요금",   _HEAT_COLOR),
]


def _per_m2_tab(df: pd.DataFrame) -> None:
    """Cost per m² — normalise billing by floor area for fair cross-tenant comparison."""
    st.subheader("Cost per m² (평당 요금 분석)")
    st.caption("단위: 만원/m² · Normalises cost by tenant floor area for fair comparison.")

    if "size_m2" not in df.columns:
        st.warning("size_m2 column not found — cannot compute per-m² metrics.")
        return

    wdf = df[df["size_m2"].notna() & (df["size_m2"] > 0)].copy()
    if wdf.empty:
        st.warning("No rows with valid size_m2 found.")
        return

    present_perm2 = [(c, lbl, clr) for c, lbl, clr in _PERM2_COLS if c in wdf.columns]
    if not present_perm2:
        st.warning("No cost columns found.")
        return

    # Compute per-m² columns
    for col, _, _ in present_perm2:
        wdf[f"{col}_pm2"] = (wdf[col].fillna(0) / wdf["size_m2"]).round(4)

    util_labels = [lbl for _, lbl, _ in present_perm2]
    sel = st.radio("Utility", util_labels, horizontal=True, key="perm2_util")
    col, lbl, clr = next(r for r in present_perm2 if r[1] == sel)
    pm2_col = f"{col}_pm2"

    _n = len(wdf)
    top_n = st.slider("Show top N brands", min(10, _n), _n, min(40, _n), key="perm2_n") if _n > 1 else _n

    sort_df = wdf.sort_values(pm2_col, ascending=False)
    plot_df = sort_df.head(top_n).iloc[::-1].copy()

    # Outlier fences
    vals = wdf[pm2_col].dropna()
    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q75 - q25
    upper_fence = q75 + 1.5 * iqr
    lower_fence = max(q25 - 1.5 * iqr, 0.0)
    median_pm2 = vals.median()
    mean_pm2   = vals.mean()

    def _bar_color(v):
        if v >= upper_fence:
            return "#C44E52"   # red — abnormally high
        if 0 < lower_fence and v <= lower_fence:
            return "#55A868"   # green — abnormally low (unusually cheap)
        return clr

    bar_colors = [_bar_color(v) for v in plot_df[pm2_col].fillna(0)]

    fig = go.Figure(go.Bar(
        y=plot_df["brand"],
        x=plot_df[pm2_col].fillna(0),
        orientation="h",
        marker_color=bar_colors,
        marker_line_color="white",
        marker_line_width=0.5,
        text=[f"{v:.4f}" for v in plot_df[pm2_col].fillna(0)],
        textposition="outside",
        textfont=dict(size=9, color="#666666"),
        customdata=plot_df[["size_m2", col]].fillna(0).values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"{lbl}/m²: %{{x:.4f}} 만원/m²<br>"
            "Floor area: %{customdata[0]:,.1f} m²<br>"
            f"{lbl}: %{{customdata[1]:,.0f}} 만원<extra></extra>"
        ),
        name=f"{lbl}/m²",
    ))

    fig.add_vline(x=float(median_pm2), line_dash="dot", line_color=_HEAT_COLOR,
                  line_width=1.5, annotation_text=f"Median {median_pm2:.4f}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=_HEAT_COLOR))
    fig.add_vline(x=float(upper_fence), line_dash="dash", line_color="#C44E52",
                  line_width=1.5, annotation_text=f"High fence {upper_fence:.4f}",
                  annotation_position="top left",
                  annotation_font=dict(size=10, color="#C44E52"))
    if lower_fence > 0:
        fig.add_vline(x=float(lower_fence), line_dash="dash", line_color="#55A868",
                      line_width=1.5, annotation_text=f"Low fence {lower_fence:.4f}",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color="#55A868"))

    max_label_len = plot_df["brand"].astype(str).str.len().max() if len(plot_df) else 20
    left_margin = min(max(max_label_len * 7, 120), 320)
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(
            text=f"<b>{lbl} per m² · red = high outlier · green = low outlier</b>",
            font=dict(size=13, color="#222222"), x=0,
        ),
        height=max(420, top_n * 22 + 100),
        xaxis=dict(title="만원 / m²", showgrid=True, gridcolor=_GRID, griddash="dot",
                   zeroline=False, tickfont=dict(size=10, color="#555555")),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=10, color="#555555"),
                   automargin=True),
        showlegend=False,
        margin=dict(l=left_margin, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary metrics
    n_high = int((wdf[pm2_col] >= upper_fence).sum())
    n_low  = int((lower_fence > 0) and (wdf[pm2_col] <= lower_fence).sum())
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Brands (with size)", len(vals))
    c2.metric("Median (만원/m²)", f"{median_pm2:.4f}")
    c3.metric("Mean (만원/m²)",   f"{mean_pm2:.4f}")
    c4.metric("High outliers",    n_high)
    c5.metric("Low outliers",     n_low)

    # Full ranked table with per-m² column + flag
    tbl = wdf.sort_values(pm2_col, ascending=False).copy()
    tbl["high_outlier"] = tbl[pm2_col] >= upper_fence
    tbl["low_outlier"]  = (lower_fence > 0) & (tbl[pm2_col] <= lower_fence)
    show_cols = ["brand", "building", "floor", "size_m2", col, pm2_col,
                 "high_outlier", "low_outlier"]
    show_cols = [c for c in show_cols if c in tbl.columns]
    out = add_display_index(tbl[show_cols].copy())
    st.markdown(f"**{len(out)} brands** with floor area — sorted by {lbl}/m² (high → low)")
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, filename=f"billing_per_m2_{lbl}.xlsx", sheet_name="per_m2")
