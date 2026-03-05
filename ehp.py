import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data import read_ehp_oac_sheet, read_ehp_raw_slice, compute_monthly_usage, group_raw_slice_by_year
from viz import plot_hist_with_tails

_PALETTE = [
    "#4C72B0", "#DD8A00", "#C44E52", "#55A868",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
_BASE_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Arial Black, Arial, sans-serif", color="#111111"),
)


@st.cache_data(show_spinner="Parsing EHP sheet...", max_entries=1)
def _cached_read(name: str, data: bytes, sheet: str):
    return read_ehp_oac_sheet(name, data, sheet)


def _build_pivot(raw_slice: pd.DataFrame) -> pd.DataFrame:
    """Sum all meter rows per month → pivot: index=월, columns=year."""
    import re as _re
    yr_pat = _re.compile(r'^(20\d{2})_(\d{1,2})월$')
    usage = compute_monthly_usage(raw_slice)
    records = []
    for col in usage.columns:
        m = yr_pat.match(str(col))
        if not m:
            continue
        yr, mo = int(m.group(1)), int(m.group(2))
        total = pd.to_numeric(usage[col], errors="coerce").sum(min_count=1)
        records.append({"year": yr, "month": mo, "total": total})
    if not records:
        return pd.DataFrame()
    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None
    return pivot


# ─── Tab 4: Per-Meter Analysis ────────────────────────────────────────────────

def _tab_meter(usage: pd.DataFrame) -> None:
    import re as _re
    yr_pat = _re.compile(r'^(20\d{2})_(\d{1,2})월$')
    if "계량기 번호" not in usage.columns:
        st.warning("계량기 번호 column not found.")
        return
    meters = sorted(usage["계량기 번호"].dropna().unique(), key=str)
    sel = st.selectbox("계량기 번호 선택", meters, key="ehp_meter_sel")
    row = usage[usage["계량기 번호"] == sel]
    records = []
    for col in usage.columns:
        m = yr_pat.match(str(col))
        if not m:
            continue
        yr, mo = int(m.group(1)), int(m.group(2))
        val = pd.to_numeric(row[col], errors="coerce").sum(min_count=1)
        records.append({"year": yr, "month": mo, "total": val})
    if not records:
        st.info("No data for this meter.")
        return
    rdf = pd.DataFrame(records)
    pivot = rdf.pivot(index="month", columns="year", values="total")
    pivot.index = [f"{m}월" for m in pivot.index]
    pivot.columns.name = None
    _tab_yearly(pivot)
    st.divider()
    _tab_month(pivot, key="ehp_month_sel_meter")


# ─── Tab 1: Yearly Usage ──────────────────────────────────────────────────────

def _tab_yearly(pivot: pd.DataFrame) -> None:
    year_totals = pivot.sum(min_count=1)
    years  = [str(y) for y in year_totals.index]
    values = year_totals.values

    fig = go.Figure(go.Bar(
        x=years, y=values,
        marker_color="#4C72B0",
        text=[f"{v:,.0f}" for v in values], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text="<b>연간 총 사용량 (kWh)</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(type="category", title=dict(text="연도", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame({"Year": years, "Total kWh": [f"{v:,.0f}" for v in values]})
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ─── Tab 2: Year Comparison ───────────────────────────────────────────────────

def _tab_compare(pivot: pd.DataFrame) -> None:
    available = list(pivot.columns)
    if len(available) < 2:
        st.warning("Need at least 2 years of data.")
        return

    c1, c2 = st.columns(2)
    with c1:
        yr_a = st.selectbox("Year A", available, index=len(available) - 2, key="ehp_cmp_a")
    with c2:
        yr_b = st.selectbox("Year B", available, index=len(available) - 1, key="ehp_cmp_b")

    if yr_a == yr_b:
        st.info("Select two different years.")
        return

    months = list(pivot.index)
    ta = pivot[yr_a].tolist()
    tb = pivot[yr_b].tolist()

    fig = go.Figure([
        go.Bar(name=str(yr_a), x=months, y=ta, marker_color=_PALETTE[0],
               hovertemplate=f"<b>{yr_a}</b> %{{x}}: %{{y:,.0f}} kWh<extra></extra>"),
        go.Bar(name=str(yr_b), x=months, y=tb, marker_color=_PALETTE[1],
               hovertemplate=f"<b>{yr_b}</b> %{{x}}: %{{y:,.0f}} kWh<extra></extra>"),
    ])
    fig.update_layout(
        **_BASE_LAYOUT,
        barmode="group",
        title=dict(text=f"<b>월별 사용량 — {yr_a} vs {yr_b}</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(title=dict(text="월", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        legend=dict(orientation="h", x=0, y=1.05, yanchor="bottom"),
        margin=dict(l=60, r=20, t=65, b=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame({
        "Month": months,
        f"{yr_a} kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in ta],
        f"{yr_b} kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in tb],
    })
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ─── Tab 3: Month Comparison ──────────────────────────────────────────────────

def _tab_month(pivot: pd.DataFrame, key: str = "ehp_month_sel") -> None:
    all_labels = [f"{m}월" for m in range(1, 13)]
    available  = [m for m in all_labels if m in pivot.index]

    sel_month = st.selectbox("Month", available, key=key)
    row    = pivot.loc[sel_month]
    years  = [str(y) for y in row.index]
    values = row.values

    fig = go.Figure(go.Bar(
        x=years, y=values,
        marker_color="#4C72B0",
        text=[f"{v:,.0f}" if pd.notna(v) else "" for v in values], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.0f} kWh<extra></extra>",
    ))
    fig.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>{sel_month} 연도별 사용량</b>", font=dict(size=14, color="#111111"), x=0),
        height=380,
        xaxis=dict(type="category", title=dict(text="연도", font=dict(color="#111111")), showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="kWh", font=dict(color="#111111")), showgrid=True, gridcolor="#AAAAAA", zeroline=False, tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    tbl = pd.DataFrame({"Year": years, "kWh": [f"{v:,.0f}" if pd.notna(v) else "" for v in values]})
    st.dataframe(tbl, hide_index=True, use_container_width=True)


# ─── Public entry point ───────────────────────────────────────────────────────

def render_ehp_view(name: str, data: bytes, sheet: str) -> None:
    st.subheader("EHP 전기 사용량 분석")

    analysis_type = st.radio("분석 유형", ["OAC", "전용 EHP"], horizontal=True, key="ehp_analysis_type")

    if analysis_type == "OAC":
        _render_oac(name, data, sheet)
    else:
        _render_ehp_dedicated(name, data, sheet)


def _render_oac(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ OAC 전기 사용량 — 단위: kWh")

    raw_slice = read_ehp_raw_slice(name, data, sheet)

    pivot = _build_pivot(raw_slice)
    if pivot.empty:
        st.error("No monthly usage data found.")
        return

    usage = compute_monthly_usage(raw_slice)

    tab1, tab2, tab3, tab4 = st.tabs(["Yearly Usage", "Year Comparison", "Month Comparison", "계량기별"])
    with tab1:
        _tab_yearly(pivot)
    with tab2:
        _tab_compare(pivot)
    with tab3:
        _tab_month(pivot)
    with tab4:
        _tab_meter(usage)

    with st.expander("Cumulative readings — grouped by year"):
        year_groups = group_raw_slice_by_year(raw_slice)
        yr_tabs = st.tabs([str(y) for y in year_groups])
        for tab, (yr, ydf) in zip(yr_tabs, year_groups.items()):
            with tab:
                st.dataframe(ydf.rename(columns=lambda c: c.replace(f"{yr}_", "")), use_container_width=True)

    with st.expander("Monthly usage — grouped by year"):
        usage_groups = group_raw_slice_by_year(usage)
        u_tabs = st.tabs([str(y) for y in usage_groups])
        for tab, (yr, ydf) in zip(u_tabs, usage_groups.items()):
            with tab:
                st.dataframe(ydf.rename(columns=lambda c: c.replace(f"{yr}_", "")), use_container_width=True)


def _render_ehp_dedicated(name: str, data: bytes, sheet: str) -> None:
    st.caption("▣ 전용 EHP 검침 자료")

    import io as _io
    _full = __import__('pandas').read_excel(_io.BytesIO(data), sheet_name=sheet, header=None, engine="calamine")
    for ri in range(len(_full)):
        if _full.iloc[ri].astype(str).str.contains("전용 EHP", na=False).any():
            _end = len(_full)
            for _rj in range(ri + 1, len(_full)):
                if _full.iloc[_rj].astype(str).str.contains("▣", na=False).any():
                    _end = _rj
                    break
            _sliced = _full.iloc[ri+1:_end, :11].reset_index(drop=True)
            _sliced.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in _sliced.iloc[0]]
            _sliced = _sliced.iloc[1:].reset_index(drop=True)
            _col0 = _sliced.columns[0]
            _sliced[_col0] = _sliced[_col0].ffill()
            _sliced = _sliced[_sliced[_col0].astype(str).str.endswith("동")].reset_index(drop=True)
            # Drop pure building-header rows (only _col0 has a value, all other columns are NaN)
            _data_cols = [c for c in _sliced.columns if c != _col0]
            _sliced = _sliced[_sliced[_data_cols].notna().any(axis=1)].reset_index(drop=True)
            if "판넬명" in _sliced.columns:
                _sliced["판넬명"] = _sliced.groupby(_col0)["판넬명"].ffill().astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
            if "장비번호" in _sliced.columns:
                _sliced["장비번호"] = _sliced.groupby(_col0)["장비번호"].ffill()

            # ── Metric selection ─────────────────────────────────────────────
            _metric_options = []
            if "전기 사용량" in _sliced.columns:
                _metric_options.append("전기 사용량")
            if "매장별 가동시간" in _sliced.columns:
                _metric_options.append("매장별 가동시간")

            if not _metric_options:
                st.warning("전기 사용량 및 매장별 가동시간 column not found.")
            else:
                _metric_label_map = {
                    "전기 사용량": ("전기 사용량 (kWh)", "kWh"),
                    "매장별 가동시간": ("가동시간 (hr)", "hr"),
                }
                all_dong  = sorted(_sliced[_col0].dropna().unique(), key=str)

                # Build chart type options based on available columns
                _chart_opts = ["건물별"]
                if "장비번호" in _sliced.columns:
                    _chart_opts.append("장비별")
                if "판넬명" in _sliced.columns:
                    _chart_opts.append("판넬별")
                if "상호" in _sliced.columns:
                    _chart_opts.append("상호별")

                chart_type = st.selectbox("차트 유형", _chart_opts, key="ehp_ded_chart")

                sel_dong = st.selectbox("동 선택", ["전체"] + all_dong, key="ehp_ded_dong")
                if sel_dong != "전체":
                    _sliced = _sliced[_sliced[_col0] == sel_dong]

                with st.expander("검색"):
                    _sc1, _sc2, _sc3 = st.columns(3)
                    with _sc1:
                        _q_panel = st.text_input("판넬명", key="ehp_search_panel")
                    with _sc2:
                        _q_jangbi = st.text_input("장비번호", key="ehp_search_jangbi")
                    with _sc3:
                        _q_sangho = st.text_input("상호", key="ehp_search_sangho")
                if _q_panel and "판넬명" in _sliced.columns:
                    _sliced = _sliced[_sliced["판넬명"].astype(str).str.contains(_q_panel, case=False, na=False)]
                if _q_jangbi and "장비번호" in _sliced.columns:
                    _sliced = _sliced[_sliced["장비번호"].astype(str).str.contains(_q_jangbi, case=False, na=False)]
                if _q_sangho and "상호" in _sliced.columns:
                    _sliced = _sliced[_sliced["상호"].astype(str).str.contains(_q_sangho, case=False, na=False)]

                _metric_tabs = st.tabs(_metric_options)
                for _mtab, metric_sel in zip(_metric_tabs, _metric_options):
                    with _mtab:
                        val_col_label, y_unit = _metric_label_map[metric_sel]
                        _ts = _sliced.copy()
                        _ts[metric_sel] = pd.to_numeric(_ts[metric_sel], errors="coerce")

                        def _bar_chart(grouped, x_labels, title, x_title, _vcl=val_col_label, _yu=y_unit):
                            grouped = grouped.sort_values(_vcl, ascending=False).reset_index(drop=True)
                            x_labels = grouped[grouped.columns[0]].tolist()
                            fig = go.Figure(go.Bar(
                                x=x_labels, y=grouped[_vcl],
                                marker_color="#4C72B0",
                                text=[f"{v:,.0f}" for v in grouped[_vcl]],
                                textposition="outside", cliponaxis=False,
                                hovertemplate=f"<b>%{{x}}</b>: %{{y:,.0f}} {_yu}<extra></extra>",
                            ))
                            fig.update_layout(
                                **_BASE_LAYOUT,
                                title=dict(text=title, font=dict(size=14, color="#111111"), x=0),
                                height=420,
                                xaxis=dict(title=dict(text=x_title, font=dict(color="#111111")), tickfont=dict(color="#111111"), showgrid=False, zeroline=False),
                                yaxis=dict(title=dict(text=_yu, font=dict(color="#111111")), tickfont=dict(color="#111111"), showgrid=True, gridcolor="#AAAAAA", zeroline=False),
                                margin=dict(l=60, r=20, t=70, b=80),
                                showlegend=False,
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(grouped, hide_index=True, use_container_width=True)

                        total_val = _ts[metric_sel].sum(min_count=1)
                        st.metric(f"합계 {val_col_label}", f"{total_val:,.0f} {y_unit}" if pd.notna(total_val) else "N/A")

                        _graph_sel = st.radio("그래프", ["바 차트", "히스토그램"], horizontal=True, key=f"ehp_graph_{metric_sel}")

                        if _graph_sel == "히스토그램":
                            import numpy as _np
                            import numpy as _np
                            _bins = st.session_state.get("bins", 50)
                            _tail = st.session_state.get("tail", 20)
                            _s    = _ts[metric_sel]
                            _v    = _s.dropna()
                            if not _v.empty:
                                _lo = float(_np.percentile(_v, _tail))
                                _hi = float(_np.percentile(_v, 100 - _tail))
                                _display_cols = [c for c in [_col0, "판넬명", "장비번호", "상호", metric_sel] if c in _ts.columns]
                                plot_hist_with_tails(
                                    _s, _bins, _lo, _hi,
                                    title=metric_sel,
                                    tail_pct=_tail,
                                    key=f"ehp_hist_{metric_sel}",
                                    source_df=_ts,
                                    val_col=metric_sel,
                                    display_cols=_display_cols,
                                )

                        else:
                            if chart_type == "건물별":
                                grouped = _ts.groupby(_col0)[metric_sel].sum().reset_index()
                                grouped.columns = [_col0, val_col_label]
                                _bar_chart(grouped, grouped[_col0].tolist(), f"<b>건물별 {metric_sel} 합계</b>", "동")
                            elif chart_type == "장비별":
                                grouped = _ts.groupby("장비번호")[metric_sel].sum().reset_index()
                                grouped.columns = ["장비번호", val_col_label]
                                _bar_chart(grouped, grouped["장비번호"].tolist(), f"<b>장비별 {metric_sel}</b>", "장비번호")
                            elif chart_type == "판넬별":
                                grouped = _ts.groupby("판넬명")[metric_sel].sum().reset_index()
                                grouped.columns = ["판넬명", val_col_label]
                                _bar_chart(grouped, grouped["판넬명"].tolist(), f"<b>판넬별 {metric_sel} 합계</b>", "판넬명")
                            elif chart_type == "상호별":
                                grouped = _ts.groupby("상호")[metric_sel].sum().reset_index()
                                grouped.columns = ["상호", val_col_label]
                                _bar_chart(grouped, grouped["상호"].tolist(), f"<b>상호별 {metric_sel}</b>", "상호")

            with st.expander("Raw data"):
                st.dataframe(_sliced, use_container_width=True)
            break
    else:
        st.error("전용 EHP 검침 자료 섹션을 찾을 수 없습니다.")


# ─── 관리비 고지서 EHP 열(냉난방) View ────────────────────────────────────────

_USAGE_METRICS = [
    ("기본요금",         "기본요금",        "원"),   # K
    ("FCU 사용량",       "FCU_사용량",      ""),     # L
    ("FCU 사용요금",     "FCU_사용요금",    "원"),   # M
    ("FCU 전용",         "FCU_전용",        "원"),   # N
    ("FCU 공용요금",     "FCU_공용요금",    "원"),   # O
    ("FCU 소계",         "FCU_소계",        "원"),   # P = K+M+O
    ("EHP 사용량 (kWh)", "EHP_사용량_kwh",  "kWh"),  # Q
    ("EHP 전용요금",     "EHP_전용요금",    "원"),   # R
]
_STRUCT_COLS = ["구분", "건물", "층수", "호수", "전용면적_m2", "공용면적_m2", "합계면적_m2", "전용면적_평", "브랜드"]


def render_ehp_billing_view(df: pd.DataFrame) -> None:
    import numpy as _np

    st.subheader("관리비 고지서 — EHP 열(냉난방)")

    all_buildings = sorted(df["건물"].dropna().unique().tolist())
    sel_building = st.selectbox("건물 선택", ["전체"] + all_buildings, key="ehpb_building")
    _df = df if sel_building == "전체" else df[df["건물"] == sel_building].copy()

    (tab_brand,) = st.tabs(["브랜드별 사용량"])

    with tab_brand:
        _render_ehpb_brand_tab(_df, _np)

    with st.expander("Raw data"):
        st.dataframe(_df, use_container_width=True)


def _render_ehpb_brand_tab(df: pd.DataFrame, _np) -> None:
    all_brands = sorted(df["브랜드"].dropna().unique().tolist())
    sel_brand = st.selectbox("브랜드 선택", all_brands, key="ehpb_brand")

    row = df[df["브랜드"] == sel_brand].iloc[0]

    # Structural info
    struct_cols = [c for c in ["건물", "층수", "호수", "전용면적_m2", "공용면적_m2", "합계면적_m2", "전용면적_평"] if c in df.columns]
    st.dataframe(
        df[df["브랜드"] == sel_brand][struct_cols].reset_index(drop=True),
        hide_index=True, use_container_width=True,
    )

    st.divider()

    def _get(col):
        if col not in df.columns:
            return None
        v = pd.to_numeric(row[col], errors="coerce")
        return float(v) if pd.notna(v) else None

    # ── Panel 1: Usage quantities ─────────────────────────────────────────────
    st.markdown("**사용량**")
    fcu_usage = _get("FCU_사용량")
    ehp_usage = _get("EHP_사용량_kwh")
    c1, c2 = st.columns(2)
    c1.metric("FCU 사용량 (m³/MWh)", f"{fcu_usage:,.2f}" if fcu_usage is not None else "N/A")
    c2.metric("EHP 사용량 (kWh)",    f"{ehp_usage:,.0f}"  if ehp_usage  is not None else "N/A")

    st.divider()

    # ── Panel 2: Monetary breakdown ───────────────────────────────────────────
    st.markdown("**요금 내역 (원)**")

    # 부과금액_전용 (S) = N + R
    # 부과금액_공용  (P) = K + M + O
    n = _get("FCU_전용")     or 0.0
    r = _get("EHP_전용요금") or 0.0
    k = _get("기본요금")     or 0.0
    m = _get("FCU_사용요금") or 0.0
    o = _get("FCU_공용요금") or 0.0

    fig2 = go.Figure()
    for name, x_label, val, color in [
        ("FCU 전용 (N)",     "부과금액 전용", n, _PALETTE[0]),
        ("EHP 전용요금 (R)",  "부과금액 전용", r, _PALETTE[1]),
        ("기본요금 (K)",     "부과금액 공용", k, _PALETTE[2]),
        ("FCU 사용요금 (M)", "부과금액 공용", m, _PALETTE[3]),
        ("FCU 공용요금 (O)", "부과금액 공용", o, _PALETTE[4]),
    ]:
        fig2.add_trace(go.Bar(
            name=name, x=[x_label], y=[val],
            marker_color=color, marker_line_color="white", marker_line_width=0.8,
            text=[f"{val:,.0f}"], textposition="inside",
        ))

    fig2.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>{sel_brand}</b> — 요금 내역", font=dict(size=13), x=0),
        barmode="stack",
        height=420,
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#111111")),
        yaxis=dict(title=dict(text="원"), showgrid=True, gridcolor="#AAAAAA",
                   zeroline=False, tickfont=dict(color="#111111")),
        margin=dict(l=60, r=20, t=70, b=60),
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
    )
    st.plotly_chart(fig2, use_container_width=True)
