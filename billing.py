from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as _stats
import streamlit as st

from data import st_safe
from features import add_display_index, download_df_as_excel, get_simple_floors, parse_floor_value
from filters import brand_search_bar
from utils import fmt_won as _fmt_won
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
    "총 합계":  ["합계", "전용", "공용"],
    "상하수도": ["합계", "전용", "공용"],
    "전기요금": ["합계", "전용", "공용"],
    "열요금":   ["합계", "냉난방 합계", "냉난방 전용", "냉난방 공용",
                 "급탕 합계", "급탕 전용", "급탕 공용"],
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

def render_hvac_view(df: pd.DataFrame) -> None:
    st.subheader("관리비 고지서 EHP 열(냉난방)")

    if df.empty:
        st.warning("열(냉난방)사용 내역 섹션을 찾을 수 없습니다.")
        return

    _hvac_analysis(df)


_LAYOUT_BASE = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(color="#000000"),
)

def _axis(grid: bool = False, **kwargs) -> dict:
    d = dict(tickfont=dict(color="#000000"), title_font=dict(color="#000000"))
    if grid:
        d.update(gridcolor="#DDDDDD", gridwidth=1)
    d.update(kwargs)
    return d


def _hvac_analysis(df: pd.DataFrame) -> None:
    # Column formula: P (소계) = K (기본요금) + M (사용요금) + O (공용요금)
    usage_col     = next((c for c in df.columns if "사용량" in c), None)
    base_col      = next((c for c in df.columns if "기본요금" in c), None)
    usage_fee_col = next((c for c in df.columns if "사용요금" in c), None)
    comm_fee_col  = next((c for c in df.columns if "공용요금" in c), None)
    amount_col    = next((c for c in df.columns if "전용" in c and "면적" not in c and "부과" not in c and "전용요금" not in c and "FCU" in c), None)
    total_col     = next((c for c in df.columns if "소계" in c), None)
    area_col      = next((c for c in df.columns if "면적" in c), None)
    numeric_cols  = [c for c in [usage_col, base_col, usage_fee_col, comm_fee_col, amount_col, total_col, area_col] if c]

    num_df = df.copy()
    for col in numeric_cols:
        num_df[col] = pd.to_numeric(
            num_df[col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    for col in [base_col, usage_fee_col, comm_fee_col]:
        if col:
            num_df[col] = num_df[col].fillna(0)

    if "브랜드" not in num_df.columns:
        st.dataframe(st_safe(df), use_container_width=True)
        return

    # ── Validate: P (소계) = K (기본요금) + M (사용요금) + O (공용요금) ────────
    _val_col = total_col or amount_col   # prefer 소계; fall back to 전용
    if _val_col and (base_col or usage_fee_col):
        actual   = num_df[_val_col].fillna(0)
        _base_s  = num_df[base_col].fillna(0)      if base_col      else pd.Series(0, index=num_df.index)
        _usage_s = num_df[usage_fee_col].fillna(0) if usage_fee_col else pd.Series(0, index=num_df.index)
        _comm_s  = num_df[comm_fee_col].fillna(0)  if comm_fee_col  else pd.Series(0, index=num_df.index)
        computed = _base_s + _usage_s + _comm_s
        diff     = actual - computed
        abs_diff = diff.abs()
        mismatch_mask = abs_diff > 1
        n_total    = len(num_df)
        n_mismatch = int(mismatch_mask.sum())
        n_match    = n_total - n_mismatch
        match_rate = n_match / n_total * 100 if n_total else 0.0

        _val_label = "소계" if total_col else "전용 합계"
        _formula   = "기본요금 + 사용요금 + 공용요금" if comm_fee_col else "기본요금 + 사용요금"
        _val_title = f"요금 정합성 검증 — {'✅ 전체 일치' if n_mismatch == 0 else f'⚠️ 불일치 {n_mismatch}건'}"
        with st.expander(_val_title, expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("검증 대상", f"{n_total}건")
            c2.metric("일치", f"{n_match}건", f"{match_rate:.1f}%")
            c3.metric("불일치", f"{n_mismatch}건")
            c4.metric("최대 차이", f"{abs_diff.max():,.0f}원" if n_total else "—")

            if n_mismatch > 0:
                show_cols = [c for c in ["브랜드", _val_col, base_col, usage_fee_col, comm_fee_col] if c and c in num_df.columns]
                mismatch_df = num_df[mismatch_mask][show_cols].copy()
                mismatch_df[f"차이 ({_val_label}−합산)"] = diff[mismatch_mask].values
                st.dataframe(st_safe(mismatch_df), use_container_width=True)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _stats_table(rows: list[tuple[str, pd.Series]], total_sum: float | None = None) -> None:
        """Render a compact stats table. rows = [(label, series), ...]
        If total_sum is provided, adds a 합계 비중 column showing each row's share of total_sum."""
        records = []
        for label, s in rows:
            s = s.replace(0, pd.NA).dropna()
            if s.empty:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            row = {
                "항목":   label,
                "업체 수": len(s),
                "합계":   f"{s.sum():,.0f}",
                "평균":   f"{s.mean():,.0f}",
                "중앙값": f"{s.median():,.0f}",
                "표준편차": f"{s.std():,.0f}",
                "최대":   f"{s.max():,.0f}",
                "최소":   f"{s.min():,.0f}",
            }
            if total_sum:
                row["합계 비중"] = f"{s.sum() / total_sum * 100:.1f}%"
            records.append(row)
        if not records:
            return
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)

        with st.expander("컬럼 설명"):
            _col_desc = [
                ("업체 수",  "해당 항목에 값이 있는(0 제외) 업체 수"),
                ("합계",     "표시된 업체들의 값을 모두 더한 총액"),
                ("평균",     "합계 ÷ 업체 수"),
                ("중앙값",   "업체를 금액 순 정렬 시 정중앙 값 — 극단값에 강건"),
                ("표준편차", "업체 간 편차 지표 — 클수록 분포 불균형"),
                ("최대/최소","가장 높은 / 낮은 업체의 값"),
            ]
            if total_sum:
                _col_desc.append(("합계 비중", f"전용 합계 총액({total_sum:,.0f}원) 대비 비율"))
            st.dataframe(pd.DataFrame(_col_desc, columns=["컬럼", "설명"]),
                         use_container_width=True, hide_index=True)

        _insight_rows = []
        for label, s in rows:
            s = s.replace(0, pd.NA).dropna()
            if s.empty or len(s) < 2:
                continue
            mean, median = s.mean(), s.median()
            cv           = s.std() / mean if mean else 0
            skew_ratio   = mean / median if median else 1
            mx, mn       = s.max(), s.min()
            irow = {
                "항목":           label,
                "분포 형태":      "▲ 우편향" if skew_ratio > 1.5 else ("▼ 좌편향" if skew_ratio < 0.67 else "≈ 균등"),
                "변동성 (CV)":    f"{cv:.2f}",
                "격차 (최대/최소)": f"{mx/mn:.0f}×" if mn > 0 else "—",
            }
            if total_sum:
                irow["합계 비중"] = f"{s.sum() / total_sum * 100:.1f}%"
            _insight_rows.append(irow)
        if _insight_rows:
            with st.expander("분포 특성"):
                st.dataframe(pd.DataFrame(_insight_rows), use_container_width=True, hide_index=True)

    def _hbar(series: pd.Series, xlab: str, color: str, key: str):
        fig = go.Figure(go.Bar(
            x=series.values,
            y=series.index.astype(str),
            orientation="h",
            marker_color=color,
            text=[f"{v:,.0f}" for v in series.values],
            textposition="auto",
            textfont=dict(color="#000000"),
        ))
        fig.update_layout(
            height=max(340, len(series) * 26),
            margin=dict(t=20, b=60, l=160, r=120),
            xaxis=_axis(grid=True, title=xlab, range=[0, series.values.max() * 1.25]),
            yaxis=_axis(autorange="reversed"),
            **_LAYOUT_BASE,
        )
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)

    def _hist(series: pd.Series, xlab: str, color: str, key: str):
        clean = series.dropna()
        if clean.empty:
            return
        x = clean.values.astype(float)
        _bins = int(st.session_state.get("bins", 10))
        _tail = float(st.session_state.get("tail", 10))
        counts, edges = np.histogram(x, bins=_bins)
        midpoints = (edges[:-1] + edges[1:]) / 2
        widths     = edges[1:] - edges[:-1]
        lo  = float(np.percentile(x, _tail))
        hi  = float(np.percentile(x, 100 - _tail))
        med = float(np.median(x))
        tail_mask   = np.array([(m <= lo or m >= hi) for m in midpoints])
        normal_mask = ~tail_mask
        xmin, xmax  = float(x.min()), float(x.max())

        _bkw = dict(
            marker_line_color="white", marker_line_width=0.8, opacity=0.9,
            textposition="outside", textfont=dict(size=9, color="#666666"),
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=midpoints[normal_mask], y=counts[normal_mask], width=widths[normal_mask],
            name="일반", marker_color="#4C72B0",
            text=[str(c) if c > 0 else "" for c in counts[normal_mask]], **_bkw,
        ))
        fig.add_trace(go.Bar(
            x=midpoints[tail_mask], y=counts[tail_mask], width=widths[tail_mask],
            name="꼬리", marker_color="#DD8A00",
            text=[str(c) if c > 0 else "" for c in counts[tail_mask]], **_bkw,
        ))
        fig.add_trace(go.Scatter(x=[None], y=[None], name="중앙값", mode="lines",
                                 line=dict(color="#C44E52", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=[None], y=[None], name=f"하위/상위 {_tail:.0f}%", mode="lines",
                                 line=dict(color="#111111", width=2, dash="dash")))

        eps = 1e-12
        if lo > xmin + eps:
            fig.add_vrect(x0=xmin, x1=lo, fillcolor="#DD8A00", opacity=0.1, line_width=0)
        if hi < xmax - eps:
            fig.add_vrect(x0=hi, x1=xmax, fillcolor="#DD8A00", opacity=0.1, line_width=0)
        fig.add_vline(x=lo,  line_dash="dash", line_color="#111111", line_width=1.5)
        fig.add_vline(x=hi,  line_dash="dash", line_color="#111111", line_width=1.5)
        fig.add_vline(x=med, line_dash="dot",  line_color="#C44E52", line_width=1.5)

        n_tail = int(np.sum(counts[tail_mask]))
        n_total = int(counts.sum())
        tail_pct_val = 100 * n_tail / n_total if n_total > 0 else 0

        fig.add_annotation(
            xref="paper", yref="paper", x=0.99, y=0.55,
            xanchor="right", yanchor="top", showarrow=False,
            text=f"하위 {_tail:.0f}%  {lo:.1f}<br>상위 {_tail:.0f}%  {hi:.1f}<br>중앙값    {med:.1f}",
            font=dict(size=11, color="#333333", family="monospace"),
            bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA",
            borderwidth=1, borderpad=6, align="left",
        )
        fig.update_layout(
            title=dict(
                text=f"<b>{xlab}</b>   <span style='font-size:12px;color:#888'>n={n_total} · 꼬리={n_tail} ({tail_pct_val:.1f}%)</span>",
                font=dict(size=13, color="#222222"), x=0,
            ),
            height=380, bargap=0,
            margin=dict(l=50, r=20, t=55, b=45),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(
                title=xlab, showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
                zeroline=False, showline=True, linecolor="#AAAAAA", linewidth=1,
                tickfont=dict(size=11, color="#222222"),
            ),
            yaxis=dict(
                title=dict(text="업체 수", font=dict(size=11, color="#222222")),
                showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
                zeroline=True, zerolinecolor="#AAAAAA", zerolinewidth=1,
                showline=True, linecolor="#AAAAAA", linewidth=1,
                rangemode="tozero", tickfont=dict(size=11, color="#222222"),
            ),
            font=dict(family="Arial, sans-serif"),
            showlegend=True,
            legend=dict(
                orientation="v", x=0.99, xanchor="right", y=0.97, yanchor="top",
                font=dict(size=11, color="#333333", family="monospace"),
                bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA", borderwidth=1,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)

    # ── Filters ───────────────────────────────────────────────────────────────
    _bldg_col  = next((c for c in num_df.columns if c in ("건물", "동", "building")), None)
    _floor_col = next((c for c in num_df.columns if "층" in c or c in ("층", "floor")), None)
    _has_gong  = num_df["브랜드"].astype(str).str.contains("공실", na=False).any()

    _fc_count = 1 + bool(_bldg_col) + bool(_floor_col)
    _fcols = st.columns(_fc_count)
    _fci = 0

    if _bldg_col:
        _all_bldg = ["전체"] + sorted(num_df[_bldg_col].dropna().unique().tolist())
        _sel_bldg = _fcols[_fci].multiselect("건물", _all_bldg, default=["전체"], key="hvac_filter_bldg")
        _fci += 1
    else:
        _sel_bldg = ["전체"]

    if _floor_col:
        _all_floors = get_simple_floors(num_df.rename(columns={_floor_col: "floor"}))
        _sel_floor = _fcols[_fci].multiselect("층", ["전체"] + _all_floors, default=["전체"], key="hvac_filter_floor")
        _fci += 1
    else:
        _sel_floor = ["전체"]

    _gong_mode = _fcols[_fci].radio(
        "공실", ["전체", "공실 제외", "공실만"],
        horizontal=True, key="hvac_filter_gong",
        disabled=not _has_gong,
    )

    # Apply filters
    if _bldg_col and "전체" not in _sel_bldg and _sel_bldg:
        num_df = num_df[num_df[_bldg_col].isin(_sel_bldg)]
    if _floor_col and "전체" not in _sel_floor and _sel_floor:
        _sel_set = set(_sel_floor)
        _floor_mask = num_df[_floor_col].apply(
            lambda v: bool(set(parse_floor_value(str(v))) & _sel_set)
        )
        num_df = num_df[_floor_mask]
    if _gong_mode == "공실만":
        num_df = num_df[num_df["브랜드"].astype(str).str.contains("공실", na=False)]
    elif _gong_mode == "공실 제외":
        num_df = num_df[~num_df["브랜드"].astype(str).str.contains("공실", na=False)]

    if num_df.empty:
        st.warning("선택한 필터 조건에 해당하는 데이터가 없습니다.")
        return

    # ── Overall statistical analysis ──────────────────────────────────────────
    brand_agg_all = num_df.groupby("브랜드")[numeric_cols].sum()

    # Top-line metrics
    st.divider()
    _mc = st.columns(4)
    _mc[0].metric("총 브랜드 수", f"{num_df['브랜드'].nunique():,}")
    if usage_col:
        _mc[1].metric("총 사용량 (Mcal)", f"{brand_agg_all[usage_col].sum():,.0f}")
    if amount_col:
        _mc[2].metric("총 전용 합계 (원)", f"{brand_agg_all[amount_col].sum():,.0f}")
    if base_col and usage_fee_col:
        _base_ratio = brand_agg_all[base_col].sum() / brand_agg_all[amount_col].sum() * 100 if amount_col and brand_agg_all[amount_col].sum() else 0
        _mc[3].metric("기본요금 비중", f"{_base_ratio:.1f}%")

    # ── 기본요금 / 사용요금 / 공용요금 비중 — stored for tab use ──────────────
    # Denominator: 소계(P) if available, else 전용 합계
    _denom_col = total_col or amount_col
    _prop_data = None
    if base_col and usage_fee_col and _denom_col:
        _prop_cols = [c for c in [base_col, usage_fee_col, comm_fee_col, amount_col, total_col] if c]
        _agg = brand_agg_all[_prop_cols].copy()
        _agg = _agg[_agg[_denom_col] > 0]
        _agg["기본요금 비중(%)"] = _agg[base_col]      / _agg[_denom_col] * 100
        _agg["사용요금 비중(%)"] = _agg[usage_fee_col] / _agg[_denom_col] * 100
        if comm_fee_col:
            _agg["공용요금 비중(%)"] = _agg[comm_fee_col] / _agg[_denom_col] * 100
        _prop_data = _agg


    # Concentration: top-5 brands share
    if amount_col:
        _total = brand_agg_all[amount_col].sum()
        _top5  = brand_agg_all[amount_col].nlargest(5)
        _top5_share = _top5.sum() / _total * 100 if _total else 0
        with st.expander(f"상위 5개 업체 집중도 — {_top5_share:.1f}%"):
            _top5_df = _top5.reset_index()
            _top5_df.columns = ["브랜드", "전용 합계 (원)"]
            _top5_df["비중"] = (_top5_df["전용 합계 (원)"] / _total * 100).map(lambda x: f"{x:.1f}%")
            _top5_df["전용 합계 (원)"] = _top5_df["전용 합계 (원)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(_top5_df, use_container_width=True, hide_index=True)

    # ── Report download ────────────────────────────────────────────────────────
    with st.expander("PDF 보고서 다운로드", expanded=False):
        st.caption("현재 데이터를 기반으로 FCU 냉난방 요금 분석 PDF 보고서를 생성합니다.")
        _hvac_lang = st.radio("언어", ["한국어 (ko)", "English (en)"],
                              horizontal=True, key="hvac_report_lang")
        if st.button("PDF 생성", key="hvac_gen_pdf"):
            from hvac_report import generate_hvac_pdf
            with st.spinner("PDF 생성 중…"):
                _pdf_bytes = generate_hvac_pdf(
                    brand_agg_all,
                    usage_col=usage_col,
                    base_col=base_col,
                    usage_fee_col=usage_fee_col,
                    comm_fee_col=comm_fee_col,
                    fee_col=total_col or amount_col,
                    area_col=area_col,
                    context={"date": __import__("datetime").date.today()},
                    lang="ko" if _hvac_lang.startswith("한") else "en",
                )
            st.session_state["hvac_pdf_bytes"] = _pdf_bytes
        if "hvac_pdf_bytes" in st.session_state:
            st.download_button(
                label="PDF 다운로드",
                data=st.session_state["hvac_pdf_bytes"],
                file_name=f"hvac_report_{__import__('datetime').date.today()}.pdf",
                mime="application/pdf",
                key="hvac_pdf_dl",
            )

    st.divider()
    # ── Brand data summary (computed once, used across tabs) ──────────────────
    ref_col = amount_col or usage_col
    brands_with_data = num_df.groupby("브랜드")[ref_col].sum().dropna() if ref_col else pd.Series(dtype=float)
    brands_with_data = brands_with_data[brands_with_data > 0]
    valid_brands = set(brands_with_data.index)
    fdf = num_df[num_df["브랜드"].isin(valid_brands)]

    # ── Anomaly flags (computed once, shared across all tabs) ─────────────────
    _anom_fee = total_col or amount_col

    def _iqr_upper(s: pd.Series) -> float:
        s = s.dropna(); s = s[s > 0]
        if len(s) < 4:
            return float("inf")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return float(q3 + 1.5 * (q3 - q1))

    _flags      = pd.DataFrame(index=brand_agg_all.index)
    _series_map: dict = {}   # flag_name → (series, threshold, unit)

    if _anom_fee and _anom_fee in brand_agg_all.columns:
        _fee_s  = brand_agg_all[_anom_fee].dropna()
        _fee_up = _iqr_upper(_fee_s)
        _flags["요금 이상치"] = (_fee_s.reindex(brand_agg_all.index) > _fee_up).fillna(False)
        _series_map["요금 이상치"] = (_fee_s, _fee_up, "원")

    if _anom_fee and area_col and _anom_fee in brand_agg_all.columns and area_col in brand_agg_all.columns:
        _a_s    = brand_agg_all[area_col].where(brand_agg_all[area_col] > 0)
        _pm2    = (brand_agg_all[_anom_fee] / _a_s).dropna()
        _pm2    = _pm2[_pm2 > 0]
        _pm2_up = _iqr_upper(_pm2)
        _flags["단위면적 이상치"] = (_pm2.reindex(brand_agg_all.index) > _pm2_up).fillna(False)
        _series_map["단위면적 이상치"] = (_pm2, _pm2_up, "원/㎡")

    if _anom_fee and usage_col and _anom_fee in brand_agg_all.columns and usage_col in brand_agg_all.columns:
        _u_s    = brand_agg_all[usage_col].where(brand_agg_all[usage_col] > 0)
        _pmc    = (brand_agg_all[_anom_fee] / _u_s).dropna()
        _pmc    = _pmc[_pmc > 0]
        _pmc_up = _iqr_upper(_pmc)
        _flags["단위사용량 이상치"] = (_pmc.reindex(brand_agg_all.index) > _pmc_up).fillna(False)
        _series_map["단위사용량 이상치"] = (_pmc, _pmc_up, "원/Mcal")

    if base_col and _anom_fee and base_col in brand_agg_all.columns and _anom_fee in brand_agg_all.columns:
        _denom_an = brand_agg_all[_anom_fee].replace(0, np.nan)
        _bp_an    = (brand_agg_all[base_col] / _denom_an * 100).dropna()
        _flags["기본요금 편중"] = (_bp_an.reindex(brand_agg_all.index) > 70).fillna(False)
        _series_map["기본요금 편중"] = (_bp_an, 70.0, "%")

    _flag_cols = list(_flags.columns)
    if _flag_cols:
        _flags["플래그 수"] = _flags[_flag_cols].sum(axis=1).astype(int)
        _flags["등급"] = _flags["플래그 수"].map(
            lambda n: "🔴 위험" if n >= 2 else ("🟠 주의" if n == 1 else "🟢 정상")
        )
    _SEVER_COLOR = {"🔴 위험": "#C44E52", "🟠 주의": "#DD8A00", "🟢 정상": "#9EBADF"}

    def _anom_badge(brand):
        """Return severity color for a brand, or None if normal."""
        if "등급" not in _flags.columns or brand not in _flags.index:
            return None
        g = _flags.at[brand, "등급"]
        return _SEVER_COLOR.get(g) if g != "🟢 정상" else None

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_rank, tab_prop, tab_fair, tab_corr, tab_anom = st.tabs(["순위", "비중", "면적별 비용 비교", "상관", "이상 탐지"])

    with tab_rank:
        if amount_col and usage_col:
            _top1_fee         = brand_agg_all[amount_col].idxmax()
            _top1_usage       = brand_agg_all[usage_col].idxmax()
            _fee_total        = brand_agg_all[amount_col].sum()
            _usage_total      = brand_agg_all[usage_col].sum()
            _top1_fee_share   = brand_agg_all.loc[_top1_fee,   amount_col] / _fee_total   * 100
            _top1_usage_share = brand_agg_all.loc[_top1_usage, usage_col]  / _usage_total * 100
            _fee_cv           = brand_agg_all[amount_col].std() / brand_agg_all[amount_col].mean()
            def _mini_metric(col, label, value, sub):
                col.markdown(
                    f"<div style='font-size:13px;color:#888;margin-bottom:4px'>{label}</div>"
                    f"<div style='font-size:20px;font-weight:600;line-height:1.3;word-break:keep-all'>{value}</div>"
                    f"<div style='font-size:13px;color:#555;margin-top:3px'>{sub}</div>",
                    unsafe_allow_html=True,
                )
            _sc = st.columns(3)
            _mini_metric(_sc[0], "최다 요금 업체",     _top1_fee,         f"{_top1_fee_share:.1f}%")
            _mini_metric(_sc[1], "최다 사용량 업체",   _top1_usage,       f"{_top1_usage_share:.1f}%")
            _mini_metric(_sc[2], "요금 변동계수 (CV)", f"{_fee_cv:.2f}",  "격차 큼" if _fee_cv > 1 else "균등")

        fee_check_cols = {k: v for k, v in [
            ("사용량 (Mcal)", usage_col), ("기본요금", base_col), ("사용요금", usage_fee_col), ("전용 합계", amount_col)
        ] if v}
        brand_fee_agg = num_df.groupby("브랜드")[list(fee_check_cols.values())].sum()
        _n_total_brands = num_df["브랜드"].nunique()
        coverage = pd.DataFrame([{
            "항목": label,
            "데이터 있는 업체": int((brand_fee_agg[col] > 0).sum()),
            "데이터 없는 업체": _n_total_brands - int((brand_fee_agg[col] > 0).sum()),
        } for label, col in fee_check_cols.items()])
        st.divider()

        _ctrl_l, _ctrl_r = st.columns([3, 2])
        with _ctrl_l:
            _chart_view_opts = (["요금 순위"] if amount_col else []) + (["사용량 순위"] if usage_col else [])
            _chart_view = st.radio("보기", _chart_view_opts, horizontal=True, key="hvac_chart_view")
        with _ctrl_r:
            top_n = st.slider("표시 업체 수", min_value=1, max_value=max(1, len(brands_with_data)),
                              value=min(20, len(brands_with_data)), step=1, key="hvac_top_n")

        with st.expander("항목별 데이터 보유 현황"):
            st.dataframe(coverage, use_container_width=True, hide_index=True)

        _stat_entries: list = []
        _stat_total: float | None = None

        if _chart_view == "요금 순위" and amount_col:
            # Fee type radio — only shown for fee view
            _fee_radio_opts = (["소계"] if total_col else ["전용 합계"]) + \
                              (["기본요금"] if base_col else []) + \
                              (["사용요금"] if usage_fee_col else []) + \
                              (["공용요금"] if comm_fee_col else [])
            fee_sel = st.radio(
                "요금 항목", _fee_radio_opts,
                horizontal=True, key="hvac_fee_sel",
            )
            top_brands = (fdf.groupby("브랜드")[amount_col].sum()
                          .dropna().sort_values(ascending=False).head(top_n).index.tolist())
            sub_fdf = fdf[fdf["브랜드"].isin(top_brands)]
            if fee_sel in ("전용 합계", "소계"):
                _base_vals  = sub_fdf.groupby("브랜드")[base_col].sum().reindex(top_brands).fillna(0)      if base_col      else pd.Series(0, index=top_brands)
                _usage_vals = sub_fdf.groupby("브랜드")[usage_fee_col].sum().reindex(top_brands).fillna(0) if usage_fee_col else pd.Series(0, index=top_brands)
                _comm_vals  = sub_fdf.groupby("브랜드")[comm_fee_col].sum().reindex(top_brands).fillna(0)  if comm_fee_col  else pd.Series(0, index=top_brands)
                _pivot_dict = {"기본요금": _base_vals, "사용요금": _usage_vals}
                if comm_fee_col:
                    _pivot_dict["공용요금"] = _comm_vals
                pivot = pd.DataFrame(_pivot_dict, index=top_brands)
                pivot = pivot.assign(_total=pivot.sum(axis=1)).sort_values("_total", ascending=False).drop(columns="_total")
                fig = go.Figure()
                _stacked_traces = [("기본요금", "#9B59B6"), ("사용요금", "#27AE60")]
                if comm_fee_col:
                    _stacked_traces.append(("공용요금", "#E67E22"))
                for label, color in _stacked_traces:
                    if label in pivot.columns:
                        _rank_ylabels = [
                            ("⛔ " if (_flags.at[b, "플래그 수"] >= 2 if b in _flags.index else False)
                             else "⚠ "  if (_flags.at[b, "플래그 수"] == 1 if b in _flags.index else False)
                             else "") + str(b)[:26]
                            for b in pivot.index
                        ] if _flag_cols else list(pivot.index.astype(str))
                        fig.add_trace(go.Bar(
                            name=label, x=pivot[label], y=_rank_ylabels,
                            orientation="h", marker_color=color,
                        ))
                totals = pivot.sum(axis=1)
                _rank_label_map = {b: lbl for b, lbl in zip(pivot.index, _rank_ylabels)} if _flag_cols else {}
                annotations = [
                    dict(
                        x=totals[brand], y=_rank_label_map.get(brand, str(brand)),
                        text=f"<b>{totals[brand]:,.0f}</b>",
                        xanchor="left", yanchor="middle",
                        showarrow=False,
                        font=dict(color="#000000", size=11),
                        xshift=6,
                    )
                    for brand in pivot.index
                ]
                fig.update_layout(
                    barmode="stack",
                    height=max(340, len(pivot) * 36),
                    margin=dict(t=20, b=60, l=160, r=160),
                    xaxis=_axis(grid=True, title="금액 (원)", range=[0, totals.max() * 1.35]),
                    yaxis=_axis(autorange="reversed"),
                    legend=dict(
                        font=dict(color="#000000"),
                        title=dict(text="구분  (<b>굵은 숫자</b> = 합계)", font=dict(color="#000000", size=11)),
                    ),
                    annotations=annotations,
                    **_LAYOUT_BASE,
                )
                st.plotly_chart(fig, use_container_width=True, key="hvac_fee_stacked", theme=None)

                _stat_entries = [("기본요금 (원)", pivot["기본요금"]), ("사용요금 (원)", pivot["사용요금"])]
                if comm_fee_col:
                    _stat_entries.append(("공용요금 (원)", pivot["공용요금"]))
                _denom_lbl = "소계 (원)" if total_col else "전용 합계 (원)"
                _stat_entries.append((_denom_lbl, totals))
                _stat_total = float(totals.sum()) or None
            else:
                if fee_sel == "기본요금":
                    fee_col_sel, color = base_col, "#9B59B6"
                elif fee_sel == "사용요금":
                    fee_col_sel, color = usage_fee_col, "#27AE60"
                else:  # 공용요금
                    fee_col_sel, color = comm_fee_col, "#E67E22"
                grp = sub_fdf.groupby("브랜드")[fee_col_sel].sum().reindex(top_brands).fillna(0)
                grp = grp.sort_values(ascending=False)
                _overall_total = sub_fdf.groupby("브랜드")[amount_col].sum().reindex(top_brands).fillna(0).sum()
                _hbar(grp, f"{fee_sel} (원)", color, "hvac_fee_single")
                _stat_entries = [(f"{fee_sel} (원)", grp)]
                _stat_total = float(_overall_total) or None

        elif _chart_view == "사용량 순위" and usage_col:
            grp_usage = (fdf.groupby("브랜드")[usage_col].sum()
                         .dropna().nlargest(top_n).sort_values(ascending=False))
            _hbar(grp_usage, "사용량 (Mcal)", _WATER_COLOR, "hvac_rank_usage")
            _stat_entries = [("사용량 (Mcal)", grp_usage)]

        with st.expander("통계"):
            if _stat_entries:
                _stats_table(_stat_entries, total_sum=_stat_total)

    with tab_prop:
        if _prop_data is not None:
            _agg             = _prop_data
            _base_pct        = _agg["기본요금 비중(%)"]
            _usage_pct       = _agg["사용요금 비중(%)"]
            _comm_pct        = _agg["공용요금 비중(%)"] if comm_fee_col else pd.Series(0, index=_agg.index)
            _denom_sum       = _agg[_denom_col].sum()
            _total_base_pct  = _agg[base_col].sum()      / _denom_sum * 100
            _total_usage_pct = _agg[usage_fee_col].sum() / _denom_sum * 100
            _total_comm_pct  = _agg[comm_fee_col].sum()  / _denom_sum * 100 if comm_fee_col else 0
            _ratio           = (_agg[base_col] / _agg[usage_fee_col].replace(0, pd.NA)).dropna()
            _ratio_pct       = _ratio * 100
            _denom_label     = "소계" if total_col else "전용 합계"

            _n_base_dom  = int((_ratio_pct > 110).sum())
            _n_usage_dom = int((_ratio_pct < 90).sum())

            # ── Precompute shared objects (used across views) ─────────────────
            _ratio_num = _ratio_pct.reindex(_agg.index)

            def _dominant_label(v):
                if pd.isna(v):  return "—"
                if v >= 200:    return "기본요금 우세 ●●●"
                if v >= 150:    return "기본요금 우세 ●●"
                if v >= 110:    return "기본요금 우세 ●"
                if v <= 50:     return "사용요금 우세 ●●●"
                if v <= 75:     return "사용요금 우세 ●●"
                if v < 90:      return "사용요금 우세 ●"
                return "동일"

            _label_colors = {
                "기본요금 우세 ●●●": "background-color: #6C3483; color: white",
                "기본요금 우세 ●●":  "background-color: #9B59B6; color: white",
                "기본요금 우세 ●":   "background-color: #C39BD3",
                "동일":             "background-color: #D5F5E3",
                "사용요금 우세 ●":   "background-color: #F1948A",
                "사용요금 우세 ●●":  "background-color: #E74C3C; color: white",
                "사용요금 우세 ●●●": "background-color: #C0392B; color: white",
            }

            _ratio_table = pd.DataFrame({
                "기본요금 (원)":       _agg[base_col].map(lambda v: f"{v:,.0f}"),
                "사용요금 (원)":       _agg[usage_fee_col].map(lambda v: f"{v:,.0f}"),
                "비율 (기본÷사용, %)": _ratio_num.map(lambda v: f"{v:.1f}%" if pd.notna(v) else "—"),
                "우세 항목":          _ratio_num.map(_dominant_label),
            }, index=_agg.index).sort_values("비율 (기본÷사용, %)", ascending=False)

            _above1_mask = _ratio_pct > 100
            _below1_mask = _ratio_pct < 100
            _equal_mask  = ~_above1_mask & ~_below1_mask

            _ref_cols_for_table = [c for c in [base_col, usage_fee_col, comm_fee_col, _denom_col] if c]
            _ref_rename_map = {
                base_col:      "기본요금 (원)",
                usage_fee_col: "사용요금 (원)",
                _denom_col:    "소계 (원)" if total_col else "전용 합계 (원)",
            }
            if comm_fee_col:
                _ref_rename_map[comm_fee_col] = "공용요금 (원)"

            # ── Summary metrics row ───────────────────────────────────────────
            _m_cols = 3 + (1 if comm_fee_col else 0)
            _mc2 = st.columns(_m_cols)
            _mc2[0].metric("기본요금 비중", f"{_total_base_pct:.1f}%")
            _mc2[1].metric("사용요금 비중", f"{_total_usage_pct:.1f}%")
            if comm_fee_col:
                _mc2[2].metric("공용요금 비중", f"{_total_comm_pct:.1f}%")
            _mc2[-1].metric("기본 우세 업체", f"{_n_base_dom}개",
                            f"사용 우세 {_n_usage_dom}개", delta_color="off")

            st.divider()

            _pv_l, _pv_r = st.columns([3, 2])
            with _pv_l:
                _prop_view = st.radio(
                    "보기", ["구성 비중", "기본÷사용 비율", "분포"],
                    horizontal=True, key="hvac_prop_view",
                )
            with _pv_r:
                if _prop_view == "구성 비중":
                    _comp_view = st.radio(
                        "구분", ["전체 구성 비중", "업체별 요금 구성 비중"],
                        horizontal=True, key="hvac_comp_view",
                    )
            st.divider()

            # ── View: 구성 비중 ───────────────────────────────────────────────
            if _prop_view == "구성 비중":
                if "hvac_comp_view" not in st.session_state:
                    st.session_state["hvac_comp_view"] = "전체 구성 비중"
                _comp_view = st.session_state.get("hvac_comp_view", "전체 구성 비중")

                if _comp_view == "전체 구성 비중":
                    # ── Donut ────────────────────────────────────────────────
                    _donut_labels = ["기본요금", "사용요금"]
                    _donut_values = [_total_base_pct, _total_usage_pct]
                    _donut_colors = ["#9B59B6", "#27AE60"]
                    if comm_fee_col and _total_comm_pct > 0:
                        _donut_labels.append("공용요금")
                        _donut_values.append(_total_comm_pct)
                        _donut_colors.append("#E67E22")
                    _fig_donut = go.Figure(go.Pie(
                        labels=_donut_labels,
                        values=_donut_values,
                        hole=0.55,
                        marker_colors=_donut_colors,
                        textinfo="label+percent",
                        textfont=dict(color="#000000"),
                        insidetextorientation="radial",
                    ))
                    _fig_donut.update_layout(
                        height=320,
                        margin=dict(t=30, b=10, l=10, r=10),
                        showlegend=False,
                        annotations=[dict(text="전체", x=0.5, y=0.5,
                                          font=dict(size=13, color="#000000"), showarrow=False)],
                        **_LAYOUT_BASE,
                    )
                    st.plotly_chart(_fig_donut, use_container_width=True, key="hvac_donut", theme=None)

                    # ── Stats table ──────────────────────────────────────────
                    _comp_rows = [("기본요금", _total_base_pct, _base_pct),
                                  ("사용요금", _total_usage_pct, _usage_pct)]
                    if comm_fee_col:
                        _comp_rows.append(("공용요금", _total_comm_pct, _comm_pct))
                    st.dataframe(pd.DataFrame([{
                        "항목":       lbl,
                        "전체 비중":  f"{total:.1f}%",
                        "업체 평균":  f"{s.mean():.1f}%",
                        "중앙값":     f"{s.median():.1f}%",
                        "표준편차":   f"{s.std():.1f}%",
                        "최대":       f"{s.max():.1f}%",
                        "최소":       f"{s.min():.1f}%",
                    } for lbl, total, s in _comp_rows]),
                    hide_index=True, use_container_width=True)

                else:  # 업체별 요금 구성 비중
                    # ── Stacked bar ──────────────────────────────────────────
                    _sorted = _agg.sort_values("기본요금 비중(%)", ascending=True)
                    _fig_prop = go.Figure()
                    _bar_traces = [("기본요금 비중(%)", "#9B59B6"), ("사용요금 비중(%)", "#27AE60")]
                    if comm_fee_col:
                        _bar_traces.append(("공용요금 비중(%)", "#E67E22"))
                    _prop_ylabels = [
                        ("⛔ " if (_flags.at[b, "플래그 수"] >= 2 if b in _flags.index else False)
                         else "⚠ "  if (_flags.at[b, "플래그 수"] == 1 if b in _flags.index else False)
                         else "") + str(b)[:26]
                        for b in _sorted.index
                    ] if _flag_cols else list(_sorted.index.astype(str))
                    for _lbl, _clr in _bar_traces:
                        _fig_prop.add_trace(go.Bar(
                            name=_lbl.replace(" 비중(%)", ""),
                            x=_sorted[_lbl],
                            y=_prop_ylabels,
                            orientation="h",
                            marker_color=_clr,
                            text=_sorted[_lbl].map(lambda v: f"{v:.0f}%"),
                            textposition="inside",
                            textfont=dict(color="white", size=10),
                        ))
                    _fig_prop.add_vline(
                        x=70, line_dash="dot", line_color="#C44E52", line_width=1.5,
                        annotation_text="편중 기준 70%",
                        annotation_position="top",
                        annotation_font=dict(size=9, color="#C44E52"),
                    )
                    _fig_prop.update_layout(
                        barmode="stack",
                        height=max(340, len(_sorted) * 26),
                        margin=dict(t=20, b=40, l=160, r=20),
                        xaxis=_axis(title="비중 (%)", range=[0, 100]),
                        yaxis=_axis(autorange="reversed"),
                        legend=dict(font=dict(color="#000000")),
                        **_LAYOUT_BASE,
                    )
                    st.plotly_chart(_fig_prop, use_container_width=True, key="hvac_prop_stack", theme=None)

                    # ── Stats table ──────────────────────────────────────────
                    _prop_stat_rows = []
                    for _plbl, _ps in [("기본요금", _base_pct), ("사용요금", _usage_pct)] + \
                                      ([("공용요금", _comm_pct)] if comm_fee_col else []):
                        _prop_stat_rows.append({
                            "항목":    _plbl,
                            "평균":    f"{_ps.mean():.1f}%",
                            "중앙값":  f"{_ps.median():.1f}%",
                            "표준편차": f"{_ps.std():.1f}%",
                            "최대":    f"{_ps.max():.1f}%",
                            "최소":    f"{_ps.min():.1f}%",
                        })
                    st.dataframe(pd.DataFrame(_prop_stat_rows), hide_index=True, use_container_width=True)

            # ── View: 기본÷사용 비율 ─────────────────────────────────────────
            elif _prop_view == "기본÷사용 비율":
                _rmc = st.columns(4)
                _rmc[0].metric("평균 비율",   f"{_ratio_pct.mean():.1f}%",
                               "전체 업체의 평균", delta_color="off")
                _rmc[1].metric("중앙값 비율", f"{_ratio_pct.median():.1f}%",
                               "상위 50% 기준선", delta_color="off")
                _rmc[2].metric("최고 비율",   f"{_ratio_pct.max():.1f}%",
                               "기본요금 부담 가장 큰 업체", delta_color="off")
                _rmc[3].metric("최저 비율",   f"{_ratio_pct.min():.1f}%",
                               "사용요금 비중 가장 큰 업체", delta_color="off")

                _ratio_sorted = _ratio_pct.sort_values(ascending=False)
                _fig_ratio = go.Figure(go.Bar(
                    x=_ratio_sorted.values,
                    y=_ratio_sorted.index.astype(str),
                    orientation="h",
                    marker_color=["#9B59B6" if v >= 100 else "#27AE60" for v in _ratio_sorted.values],
                    text=[f"{v:.1f}%" for v in _ratio_sorted.values],
                    textposition="outside",
                    textfont=dict(color="#000000"),
                ))
                _fig_ratio.add_vline(
                    x=100, line_dash="solid", line_color="#C44E52", line_width=2,
                    annotation_text="<b>100% 기준선 (1:1)</b>",
                    annotation_position="top",
                    annotation_font=dict(color="#FFFFFF", size=12),
                    annotation_bgcolor="#C44E52",
                    annotation_bordercolor="#C44E52",
                    annotation_borderwidth=1,
                    annotation_borderpad=4,
                )
                _fig_ratio.update_layout(
                    height=max(340, len(_ratio_sorted) * 26),
                    margin=dict(t=20, b=40, l=160, r=100),
                    xaxis=_axis(grid=True, title="기본요금 ÷ 사용요금 (%)"),
                    yaxis=_axis(autorange="reversed"),
                    **_LAYOUT_BASE,
                )
                st.plotly_chart(_fig_ratio, use_container_width=True, key="hvac_ratio_bar", theme=None)

            # ── View: 분포 ────────────────────────────────────────────────────
            elif _prop_view == "분포":
                _fee_hist_opts = [("기본요금 비중", _base_pct, "#9B59B6"),
                                  ("사용요금 비중", _usage_pct, "#27AE60")] + \
                                 ([("공용요금 비중", _comm_pct, "#E67E22")] if comm_fee_col else [])
                _hist_sel = st.radio("항목", [l for l, _, _ in _fee_hist_opts],
                                     horizontal=True, key="hvac_pct_hist_sel")
                _hist_s, _hist_clr = next((s, c) for l, s, c in _fee_hist_opts if l == _hist_sel)
                _hist(_hist_s, f"{_hist_sel} (%)", _hist_clr, "hvac_pct_hist")

                # ── Stats table ───────────────────────────────────────────────
                _hs = _hist_s.dropna()
                _h_tail = float(st.session_state.get("tail", 10))
                _h_lo   = float(np.percentile(_hs, _h_tail))
                _h_hi   = float(np.percentile(_hs, 100 - _h_tail))
                st.dataframe(pd.DataFrame([{
                    "n":    len(_hs),
                    "최솟값":  f"{_hs.min():.1f}%",
                    f"하위 {_h_tail:.0f}%": f"{_h_lo:.1f}%",
                    "중앙값":  f"{_hs.median():.1f}%",
                    "평균":    f"{_hs.mean():.1f}%",
                    "표준편차": f"{_hs.std():.1f}%",
                    f"상위 {_h_tail:.0f}%": f"{_h_hi:.1f}%",
                    "최댓값":  f"{_hs.max():.1f}%",
                }]), hide_index=True, use_container_width=True)

                # ── Tail brand tables ─────────────────────────────────────────
                _tail_low  = _hs[_hs <= _h_lo]
                _tail_high = _hs[_hs >= _h_hi]
                _middle    = _hs[(_hs > _h_lo) & (_hs < _h_hi)].sort_values(ascending=False)
                _tail_cols_raw = [c for c in [base_col, usage_fee_col, comm_fee_col, _denom_col] if c]
                _tail_rename   = {
                    base_col:      "기본요금 (원)",
                    usage_fee_col: "사용요금 (원)",
                    _denom_col:    "소계 (원)" if total_col else "전용 합계 (원)",
                }
                if comm_fee_col:
                    _tail_rename[comm_fee_col] = "공용요금 (원)"

                def _tail_table(mask_s: pd.Series, caption: str):
                    if mask_s.empty:
                        return
                    st.caption(caption)
                    _pct_lbl = f"{_hist_sel} (%)"
                    _t = _agg.loc[mask_s.index, _tail_cols_raw].copy().rename(columns=_tail_rename)
                    for _c in _t.columns:
                        _t[_c] = _t[_c].map(lambda v: f"{v:,.0f}")
                    _t.insert(0, _pct_lbl, mask_s.map(lambda v: f"{v:.1f}%"))
                    st.dataframe(st_safe(_t.sort_values(_pct_lbl)), use_container_width=True)

                with st.expander("구간별 업체 상세"):
                    _tail_table(_tail_high.sort_values(ascending=False),
                                f"상위 꼬리 — {_hist_sel} {_h_hi:.1f}% 초과 ({len(_tail_high)}개 업체)")
                    _tail_table(_middle.sort_values(ascending=False),
                                f"중간 구간 — {_h_lo:.1f}% ~ {_h_hi:.1f}% ({len(_middle)}개 업체)")
                    _tail_table(_tail_low.sort_values(),
                                f"하위 꼬리 — {_hist_sel} {_h_lo:.1f}% 미만 ({len(_tail_low)}개 업체)")

            # ── Tables in expanders (always shown) ────────────────────────────
            st.divider()
            with st.expander("비중 분포 요약"):
                _pct_stat_rows = []
                for _fee_lbl, _fee_s in [("기본요금", _base_pct), ("사용요금", _usage_pct)] + \
                                        ([("공용요금", _comm_pct)] if comm_fee_col else []):
                    _pct_stat_rows.append({
                        "항목": _fee_lbl,
                        "평균":   f"{_fee_s.mean():.1f}%",
                        "중앙값": f"{_fee_s.median():.1f}%",
                        "최대":   f"{_fee_s.max():.1f}%",
                        "최소":   f"{_fee_s.min():.1f}%",
                    })
                st.dataframe(pd.DataFrame(_pct_stat_rows), use_container_width=True, hide_index=True)

            with st.expander("업체별 비중 상세"):
                _pct_cols_raw  = [c for c in [base_col, usage_fee_col, comm_fee_col, _denom_col] if c]
                _pct_cols_lbl  = ["기본요금 (원)", "사용요금 (원)"] + \
                                 (["공용요금 (원)"] if comm_fee_col else []) + \
                                 [f"{_denom_label} (원)"]
                _pct_bpct_cols = ["기본요금 비중(%)", "사용요금 비중(%)"] + \
                                 (["공용요금 비중(%)"] if comm_fee_col else [])
                _pct_table = _agg[_pct_cols_raw + _pct_bpct_cols].copy()
                _pct_table.columns = _pct_cols_lbl + _pct_bpct_cols
                for _c in _pct_cols_lbl:
                    _pct_table[_c] = _pct_table[_c].map(lambda v: f"{v:,.0f}")
                for _c in _pct_bpct_cols:
                    _pct_table[_c] = _pct_table[_c].map(lambda v: f"{v:.1f}%")
                st.dataframe(st_safe(_pct_table.sort_values("기본요금 비중(%)", ascending=False)),
                             use_container_width=True)

            with st.expander("업체별 비율 상세"):
                st.dataframe(
                    _ratio_table.style.applymap(
                        lambda v: _label_colors.get(v, ""),
                        subset=["우세 항목"],
                    ),
                    use_container_width=True,
                )
                for _mask, _lbl in [(_above1_mask, f"기본 > 사용 — {int(_above1_mask.sum())}개 업체"),
                                    (_below1_mask, f"사용 > 기본 — {int(_below1_mask.sum())}개 업체"),
                                    (_equal_mask,  f"동일 — {int(_equal_mask.sum())}개 업체")]:
                    if _mask.any():
                        st.caption(_lbl)
                        _t = _agg.loc[_mask.index[_mask], [base_col, usage_fee_col]].copy()
                        _t.columns = ["기본요금 (원)", "사용요금 (원)"]
                        _t["비율 (%)"] = _ratio_pct[_mask].map(lambda v: f"{v:.1f}%")
                        for _c in ["기본요금 (원)", "사용요금 (원)"]:
                            _t[_c] = _t[_c].map(lambda v: f"{v:,.0f}")
                        st.dataframe(st_safe(_t), use_container_width=True)

            _high_base = _agg[_agg["기본요금 비중(%)"] > 80][_ref_cols_for_table + ["기본요금 비중(%)"]].copy()
            _no_base   = _agg[_agg[base_col] == 0][_ref_cols_for_table].copy() if base_col else pd.DataFrame()
            if not _high_base.empty or not _no_base.empty:
                with st.expander("심층 분석"):
                    if not _high_base.empty:
                        st.caption(f"기본요금 비중 > 80% — {len(_high_base)}개 업체")
                        _hb = _high_base.rename(columns=_ref_rename_map).copy()
                        for _c in [c for c in _hb.columns if c != "기본요금 비중(%)"]:
                            _hb[_c] = _hb[_c].map(lambda v: f"{v:,.0f}")
                        _hb["기본요금 비중(%)"] = _hb["기본요금 비중(%)"].map(lambda v: f"{v:.1f}%")
                        st.dataframe(st_safe(_hb.sort_values("기본요금 비중(%)", ascending=False)),
                                     use_container_width=True)
                    if not _no_base.empty:
                        st.caption(f"기본요금 = 0 — {len(_no_base)}개 업체")
                        _nb = _no_base.rename(columns=_ref_rename_map).copy()
                        for _c in _nb.columns:
                            _nb[_c] = _nb[_c].map(lambda v: f"{v:,.0f}")
                        st.dataframe(st_safe(_nb), use_container_width=True)

        else:
            st.info("기본요금 또는 사용요금 컬럼이 없어 비중 분석을 표시할 수 없습니다.")

    with tab_fair:
        _fee_col = total_col or amount_col
        if not _fee_col:
            st.info("소계 또는 전용 합계 컬럼이 없어 비용 비교 분석을 표시할 수 없습니다.")
        else:
            _fv_l, _fv_r = st.columns([3, 5])
            with _fv_l:
                _fair_view = st.radio(
                    "분석 기준", ["단위면적당 요금 (원/㎡·평)", "단위사용량당 요금 (원/Mcal)"],
                    horizontal=False, key="hvac_fair_view",
                )
            with _fv_r:
                st.caption(
                    "면적·사용량 규모를 보정해 업체 간 부담 수준을 동일 기준으로 나란히 비교합니다. "
                    "이상치 업체는 IQR 기준으로 표시됩니다."
                )
            st.divider()

            # ── IQR outlier helper ────────────────────────────────────────────
            def _fair_chart_and_analysis(series: pd.Series, unit: str, key_sfx: str):
                s = series.dropna()
                s = s[s > 0]
                if s.empty:
                    st.info("계산 가능한 데이터가 없습니다.")
                    return

                q25, q75 = s.quantile(0.25), s.quantile(0.75)
                iqr       = q75 - q25
                upper     = q75 + 1.5 * iqr
                lower     = max(q25 - 1.5 * iqr, 0.0)
                med       = s.median()
                mean      = s.mean()
                cv        = s.std() / mean if mean else 0

                # ── Shared outlier table helper ───────────────────────────────
                _detail_cols = [c for c in [base_col, usage_fee_col, comm_fee_col, _fee_col, usage_col, area_col] if c]
                _detail_rename = {
                    base_col:      "기본요금 (원)",
                    usage_fee_col: "사용요금 (원)",
                    _fee_col:      "소계 (원)" if total_col else "전용 합계 (원)",
                    usage_col:     "사용량 (Mcal)",
                    area_col:      "면적 (㎡)",
                }
                if comm_fee_col:
                    _detail_rename[comm_fee_col] = "공용요금 (원)"

                def _outlier_table(idx, caption):
                    if idx.empty:
                        return
                    st.caption(caption)
                    _ot = brand_agg_all.loc[idx, [c for c in _detail_cols if c in brand_agg_all.columns]].copy()
                    _ot = _ot.rename(columns={k: v for k, v in _detail_rename.items() if k in _ot.columns})
                    _ot.insert(0, unit, s[idx].map(lambda v: f"{v:,.2f}"))
                    for c in _ot.columns[1:]:
                        _ot[c] = pd.to_numeric(_ot[c], errors="coerce").map(
                            lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
                        )
                    st.dataframe(st_safe(_ot.sort_values(unit, ascending=False)), use_container_width=True)

                _chart_type = st.radio(
                    "차트 유형", ["분포 히스토그램", "업체별 막대"],
                    horizontal=True, key=f"hvac_fair_charttype_{key_sfx}",
                )

                if _chart_type == "업체별 막대":
                    def _color(v):
                        if v > upper:             return "#C44E52"
                        if lower > 0 and v < lower: return "#55A868"
                        return "#4C72B0"

                    s_sorted = s.sort_values(ascending=False)
                    colors = [_color(v) for v in s_sorted.values]

                    _fair_border_clr = ["#8B1A1A" if v > upper else ("white" if lower == 0 or v >= lower else "#1A5C2A") for v in s_sorted.values]
                    _fair_border_w   = [2.5 if v > upper else (2.5 if lower > 0 and v < lower else 0) for v in s_sorted.values]
                    fig = go.Figure(go.Bar(
                        x=s_sorted.values,
                        y=s_sorted.index.astype(str),
                        orientation="h",
                        marker_color=colors,
                        marker_line=dict(color=_fair_border_clr, width=_fair_border_w),
                        text=[f"{v:,.1f}" for v in s_sorted.values],
                        textposition="outside",
                        textfont=dict(color="#000000", size=10),
                    ))
                    fig.add_vline(x=med,  line_dash="dot",  line_color="#C44E52", line_width=2,
                                  annotation_text=f"<b>중앙값 {med:,.1f}</b>",
                                  annotation_position="top right",
                                  annotation_font=dict(color="#FFFFFF", size=11),
                                  annotation_bgcolor="#C44E52",
                                  annotation_bordercolor="#C44E52",
                                  annotation_borderwidth=1, annotation_borderpad=3)
                    fig.add_vline(x=upper, line_dash="dash", line_color="#C44E52", line_width=1.5,
                                  annotation_text=f"상한 {upper:,.1f}",
                                  annotation_position="bottom right",
                                  annotation_font=dict(color="#C44E52", size=10))
                    if lower > 0:
                        fig.add_vline(x=lower, line_dash="dash", line_color="#55A868", line_width=1.5,
                                      annotation_text=f"하한 {lower:,.1f}",
                                      annotation_position="bottom right",
                                      annotation_font=dict(color="#55A868", size=10))
                    fig.update_layout(
                        height=max(340, len(s_sorted) * 26),
                        margin=dict(t=20, b=50, l=160, r=120),
                        xaxis=_axis(grid=True, title=unit),
                        yaxis=_axis(autorange="reversed"),
                        **_LAYOUT_BASE,
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"hvac_fair_{key_sfx}", theme=None)

                    # ── Stats + expanders for bar chart (IQR) ─────────────────
                    st.dataframe(pd.DataFrame([{
                        "n": len(s), "최솟값": f"{s.min():,.2f}", "Q1": f"{q25:,.2f}",
                        "중앙값": f"{med:,.2f}", "평균": f"{mean:,.2f}", "Q3": f"{q75:,.2f}",
                        "최댓값": f"{s.max():,.2f}", "IQR": f"{iqr:,.2f}",
                        "상한 (Q3+1.5×IQR)": f"{upper:,.2f}",
                    }]), hide_index=True, use_container_width=True)

                    _over_b   = s[s > upper]
                    _under_b  = s[s < lower] if lower > 0 else pd.Series(dtype=float)
                    _lo_mask_b = (s >= lower) if lower > 0 else pd.Series(True, index=s.index)
                    _normal_b = s[(s <= upper) & _lo_mask_b]

                    with st.expander("구간별 업체 상세"):
                        _outlier_table(_over_b.index,   f"상위 이상치 — 상한({upper:,.1f}) 초과 ({len(_over_b)}개 업체)")
                        _outlier_table(_normal_b.index, f"정상 범위 ({len(_normal_b)}개 업체)")
                        if not _under_b.empty:
                            _outlier_table(_under_b.index, f"하위 이상치 — 하한({lower:,.1f}) 미만 ({len(_under_b)}개 업체)")

                else:  # 분포 히스토그램
                    x      = s.values.astype(float)
                    _bins  = int(st.session_state.get("bins", 10))
                    _tail  = float(st.session_state.get("tail", 10))
                    counts, edges = np.histogram(x, bins=_bins)
                    midpoints = (edges[:-1] + edges[1:]) / 2
                    widths    = edges[1:] - edges[:-1]
                    lo_h = float(np.percentile(x, _tail))
                    hi_h = float(np.percentile(x, 100 - _tail))
                    xmin, xmax = float(x.min()), float(x.max())
                    tail_mask   = np.array([(m <= lo_h or m >= hi_h) for m in midpoints])
                    normal_mask = ~tail_mask

                    _bkw = dict(
                        marker_line_color="white", marker_line_width=0.8, opacity=0.9,
                        textposition="outside", textfont=dict(size=9, color="#666666"),
                    )
                    hfig = go.Figure()
                    hfig.add_trace(go.Bar(
                        x=midpoints[normal_mask], y=counts[normal_mask], width=widths[normal_mask],
                        name="일반", marker_color="#4C72B0",
                        text=[str(c) if c > 0 else "" for c in counts[normal_mask]], **_bkw,
                    ))
                    hfig.add_trace(go.Bar(
                        x=midpoints[tail_mask], y=counts[tail_mask], width=widths[tail_mask],
                        name="꼬리", marker_color="#DD8A00",
                        text=[str(c) if c > 0 else "" for c in counts[tail_mask]], **_bkw,
                    ))
                    hfig.add_trace(go.Scatter(x=[None], y=[None], name="중앙값", mode="lines",
                                             line=dict(color="#C44E52", width=2, dash="dot")))
                    hfig.add_trace(go.Scatter(x=[None], y=[None], name=f"하위/상위 {_tail:.0f}%", mode="lines",
                                             line=dict(color="#111111", width=2, dash="dash")))
                    eps = 1e-12
                    if lo_h > xmin + eps:
                        hfig.add_vrect(x0=xmin, x1=lo_h, fillcolor="#DD8A00", opacity=0.1, line_width=0)
                    if hi_h < xmax - eps:
                        hfig.add_vrect(x0=hi_h, x1=xmax, fillcolor="#DD8A00", opacity=0.1, line_width=0)
                    hfig.add_vline(x=lo_h, line_dash="dash", line_color="#111111", line_width=1.5)
                    hfig.add_vline(x=hi_h, line_dash="dash", line_color="#111111", line_width=1.5)
                    hfig.add_vline(x=med,  line_dash="dot",  line_color="#C44E52", line_width=1.5)
                    n_tail_h = int(np.sum(counts[tail_mask]))
                    n_total_h = int(counts.sum())
                    tail_pct_h = 100 * n_tail_h / n_total_h if n_total_h > 0 else 0
                    hfig.add_annotation(
                        xref="paper", yref="paper", x=0.99, y=0.55,
                        xanchor="right", yanchor="top", showarrow=False,
                        text=f"하위 {_tail:.0f}%  {lo_h:.1f}<br>상위 {_tail:.0f}%  {hi_h:.1f}<br>중앙값    {med:.1f}",
                        font=dict(size=11, color="#333333", family="monospace"),
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA",
                        borderwidth=1, borderpad=6, align="left",
                    )
                    hfig.update_layout(
                        title=dict(
                            text=f"<b>{unit} 분포</b>   <span style='font-size:12px;color:#888'>n={n_total_h} · 꼬리={n_tail_h} ({tail_pct_h:.1f}%)</span>",
                            font=dict(size=13, color="#222222"), x=0,
                        ),
                        height=380, bargap=0,
                        margin=dict(l=50, r=20, t=55, b=45),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(
                            title=unit, showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
                            zeroline=False, showline=True, linecolor="#AAAAAA", linewidth=1,
                            tickfont=dict(size=11, color="#222222"),
                        ),
                        yaxis=dict(
                            title=dict(text="업체 수", font=dict(size=11, color="#222222")),
                            showgrid=True, gridcolor="#DDDDDD", gridwidth=1, griddash="dot",
                            zeroline=True, zerolinecolor="#AAAAAA", zerolinewidth=1,
                            showline=True, linecolor="#AAAAAA", linewidth=1,
                            rangemode="tozero", tickfont=dict(size=11, color="#222222"),
                        ),
                        font=dict(family="Arial, sans-serif"),
                        showlegend=True,
                        legend=dict(
                            orientation="v", x=0.99, xanchor="right", y=0.97, yanchor="top",
                            font=dict(size=11, color="#333333", family="monospace"),
                            bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA", borderwidth=1,
                        ),
                    )
                    st.plotly_chart(hfig, use_container_width=True, key=f"hvac_fair_hist_{key_sfx}", theme=None)

                    # ── Stats + expanders for histogram (tail %) ──────────────
                    st.dataframe(pd.DataFrame([{
                        "n": len(s), "최솟값": f"{s.min():,.2f}",
                        f"하위 {_tail:.0f}%": f"{lo_h:,.2f}", "중앙값": f"{med:,.2f}",
                        "평균": f"{mean:,.2f}", f"상위 {_tail:.0f}%": f"{hi_h:,.2f}",
                        "최댓값": f"{s.max():,.2f}",
                    }]), hide_index=True, use_container_width=True)

                    _tail_high_h = s[s >= hi_h]
                    _tail_low_h  = s[s <= lo_h]
                    _middle_h    = s[(s > lo_h) & (s < hi_h)]

                    with st.expander("구간별 업체 상세"):
                        _outlier_table(_tail_high_h.index, f"상위 꼬리 — {hi_h:,.1f} 초과 ({len(_tail_high_h)}개 업체)")
                        _outlier_table(_middle_h.index,    f"중간 구간 ({len(_middle_h)}개 업체)")
                        _outlier_table(_tail_low_h.index,  f"하위 꼬리 — {lo_h:,.1f} 이하 ({len(_tail_low_h)}개 업체)")


            # ── Per-area analysis ─────────────────────────────────────────────
            if _fair_view == "단위면적당 요금 (원/㎡·평)":
                if not area_col:
                    st.warning("면적 컬럼을 찾을 수 없습니다. 단위면적당 요금 분석을 사용할 수 없습니다.")
                else:
                    _area_unit = st.radio("면적 단위", ["㎡", "평"], horizontal=True, key="hvac_area_unit")
                    _M2_PER_PYEONG = 3.305785
                    _area_agg  = brand_agg_all[area_col].where(brand_agg_all[area_col] > 0)
                    if _area_unit == "평":
                        _area_agg = _area_agg / _M2_PER_PYEONG
                    _fee_agg   = brand_agg_all[_fee_col]
                    _per_area  = (_fee_agg / _area_agg).dropna()
                    _per_area.name = "단위면적당 요금"
                    _fair_chart_and_analysis(_per_area, f"원/{_area_unit}", f"perm2_{_area_unit}")

            # ── Per-usage analysis ────────────────────────────────────────────
            elif _fair_view == "단위사용량당 요금 (원/Mcal)":
                if not usage_col:
                    st.warning("사용량 컬럼을 찾을 수 없습니다. 단위사용량당 요금 분석을 사용할 수 없습니다.")
                else:
                    _usage_agg  = brand_agg_all[usage_col].where(brand_agg_all[usage_col] > 0)
                    _fee_agg    = brand_agg_all[_fee_col]
                    _per_usage  = (_fee_agg / _usage_agg).dropna()
                    _per_usage.name = "단위사용량당 요금"
                    _fair_chart_and_analysis(_per_usage, "원/Mcal", "perusage")


    with tab_corr:
        _M2_PER_PYEONG = 3.305785
        _fee_col_c = total_col or amount_col

        # ── Build analysis DataFrame (brand-level) ────────────────────────────
        _corr_base = brand_agg_all.copy()
        _corr_cols: dict[str, pd.Series] = {}

        if area_col:
            _corr_cols["면적_㎡"]   = _corr_base[area_col].where(_corr_base[area_col] > 0)
            _corr_cols["면적_평"]   = _corr_cols["면적_㎡"] / _M2_PER_PYEONG
        if usage_col:
            _corr_cols["사용량"]    = _corr_base[usage_col].where(_corr_base[usage_col] > 0)
        if base_col:
            _corr_cols["기본요금"]  = _corr_base[base_col]
        if usage_fee_col:
            _corr_cols["사용요금"]  = _corr_base[usage_fee_col]
        if comm_fee_col:
            _corr_cols["공용요금"]  = _corr_base[comm_fee_col]
        if _fee_col_c:
            _corr_cols["소계"]      = _corr_base[_fee_col_c]
        if area_col and _fee_col_c:
            _a = _corr_cols.get("면적_㎡", pd.Series(dtype=float))
            _corr_cols["단위면적요금_㎡"] = (_corr_base[_fee_col_c] / _a).replace([np.inf, -np.inf], np.nan)
            _corr_cols["단위면적요금_평"] = _corr_cols["단위면적요금_㎡"] * _M2_PER_PYEONG
        if usage_col and _fee_col_c:
            _u = _corr_cols.get("사용량", pd.Series(dtype=float))
            _corr_cols["단위사용량요금"] = (_corr_base[_fee_col_c] / _u).replace([np.inf, -np.inf], np.nan)

        _corr_frame = pd.DataFrame(_corr_cols, index=_corr_base.index)
        _col_names  = list(_corr_cols.keys())

        _COL_KO = {c: c for c in _col_names}  # already Korean

        if len(_col_names) < 2:
            st.info("상관 분석에 필요한 컬럼이 부족합니다.")
        else:
            # ── Auto-discovery ────────────────────────────────────────────────
            st.subheader("자동 상관 탐색")
            st.caption("모든 항목 쌍의 피어슨 r을 자동 계산합니다. 행을 클릭하면 아래 산점도에 반영됩니다.")
            _disc_rows = []
            for _i, _ca in enumerate(_col_names):
                for _cb in _col_names[_i + 1:]:
                    _dp = _corr_frame[[_ca, _cb]].dropna()
                    if len(_dp) < 4:
                        continue
                    _rv, _pv = _stats.pearsonr(_dp[_ca].values, _dp[_cb].values)
                    _disc_rows.append({
                        "X": _ca, "Y": _cb,
                        "r":       round(_rv, 3),
                        "R²":      round(_rv ** 2, 3),
                        "p-value": round(_pv, 4),
                        "n":       len(_dp),
                        "방향":    "양(+)" if _rv > 0 else "음(−)",
                        "강도":    ("강함" if abs(_rv) >= 0.6 else "보통" if abs(_rv) >= 0.35 else "약함"),
                    })

            if _disc_rows:
                _disc_df = pd.DataFrame(_disc_rows).sort_values("R²", ascending=False).reset_index(drop=True)

                _fc1, _fc2, _fc3 = st.columns([2, 2, 3])
                with _fc1:
                    _min_r2 = st.slider("최소 R²", 0.0, 1.0, 0.05, 0.05, key="hvac_corr_min_r2")
                with _fc2:
                    _show_ns = st.checkbox("p ≥ 0.05 포함", value=False, key="hvac_corr_nonsig")
                with _fc3:
                    _str_filter = st.multiselect("강도 필터", ["강함", "보통", "약함"],
                                                  default=["강함", "보통"], key="hvac_corr_strength")

                _shown = _disc_df[
                    (_disc_df["R²"] >= _min_r2) &
                    (_disc_df["강도"].isin(_str_filter)) &
                    (_show_ns | (_disc_df["p-value"] < 0.05))
                ].reset_index(drop=True)

                if _shown.empty:
                    st.info("현재 필터 조건에 맞는 쌍이 없습니다.")
                else:
                    st.caption(f"{len(_shown)}개 쌍 · 행 클릭 시 아래 산점도에 자동 반영됩니다")
                    _disc_event = st.dataframe(
                        _shown, hide_index=True, use_container_width=True,
                        on_select="rerun", selection_mode="single-row",
                        column_config={
                            "r":       st.column_config.NumberColumn("r",   format="%.3f"),
                            "R²":      st.column_config.NumberColumn("R²",  format="%.3f"),
                            "p-value": st.column_config.NumberColumn("p",   format="%.4f"),
                        },
                    )
                    _sel_rows = (
                        _disc_event.selection.rows
                        if _disc_event and hasattr(_disc_event, "selection") else []
                    )
                    if _sel_rows:
                        _sel = _shown.iloc[_sel_rows[0]]
                        st.session_state["hvac_corr_x"] = _sel["X"]
                        st.session_state["hvac_corr_y"] = _sel["Y"]
                        st.success(f"**{_sel['X']}** vs **{_sel['Y']}** 로드됨 (R² = {_sel['R²']:.3f})")

            # ── Correlation heatmap ───────────────────────────────────────────
            _hm_data = _corr_frame.dropna(how="all")
            if len(_hm_data) >= 3 and len(_col_names) >= 3:
                _corr_mat = _hm_data.corr()
                _hm_fig = px.imshow(
                    _corr_mat, x=_col_names, y=_col_names,
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=".2f", aspect="auto",
                    title="상관 행렬",
                )
                _hm_fig.update_layout(
                    height=420, margin=dict(l=10, r=10, t=50, b=10),
                    coloraxis_colorbar=dict(title="r"),
                    font=dict(size=11, color="#222222"),
                )
                _hm_fig.update_traces(textfont_size=10)
                st.plotly_chart(_hm_fig, use_container_width=True, key="hvac_heatmap", theme=None)

            st.divider()
            st.subheader("수동 산점도")

            _sc1, _sc2, _sc3, _sc4 = st.columns([3, 3, 1, 2])
            with _sc1:
                _x_col = st.selectbox("X축", _col_names,
                                       index=_col_names.index(st.session_state.get("hvac_corr_x", _col_names[0])),
                                       key="hvac_corr_x")
            with _sc2:
                _y_default = st.session_state.get("hvac_corr_y", _col_names[min(1, len(_col_names)-1)])
                _y_col = st.selectbox("Y축", _col_names,
                                       index=_col_names.index(_y_default),
                                       key="hvac_corr_y")
            with _sc3:
                st.write("")
                _log_x = st.checkbox("Log X", value=False, key="hvac_corr_log_x")
                _log_y = st.checkbox("Log Y", value=False, key="hvac_corr_log_y")
            with _sc4:
                st.write("")
                _rm_out = st.checkbox("이상치 제거 (IQR)", value=False, key="hvac_corr_rm_out")
                if _rm_out:
                    _iqr_k = st.slider("IQR 배율", 0.5, 3.0, 1.5, 0.1, key="hvac_corr_iqr_k")

            if not _rm_out:
                _iqr_k = 1.5

            if _x_col == _y_col:
                st.info("X축과 Y축에 서로 다른 항목을 선택하세요.")
            else:
                _sc_df = _corr_frame[[_x_col, _y_col]].copy()
                _sc_df["브랜드"] = _corr_frame.index.astype(str)
                _sc_df = _sc_df.dropna(subset=[_x_col, _y_col])

                if _rm_out:
                    for _c in [_x_col, _y_col]:
                        _q1, _q3 = _sc_df[_c].quantile(0.25), _sc_df[_c].quantile(0.75)
                        _iq = _q3 - _q1
                        _sc_df = _sc_df[(_sc_df[_c] >= _q1 - _iqr_k * _iq) & (_sc_df[_c] <= _q3 + _iqr_k * _iq)]

                if _sc_df.empty:
                    st.warning("유효한 데이터가 없습니다.")
                else:
                    _xv = np.log10(_sc_df[_x_col].values.astype(float)) if _log_x else _sc_df[_x_col].values.astype(float)
                    _yv = np.log10(_sc_df[_y_col].values.astype(float)) if _log_y else _sc_df[_y_col].values.astype(float)
                    _mask = np.isfinite(_xv) & np.isfinite(_yv)
                    _xv, _yv = _xv[_mask], _yv[_mask]

                    _sfig = px.scatter(
                        _sc_df, x=_x_col, y=_y_col, color="브랜드",
                        hover_name="브랜드",
                        log_x=_log_x, log_y=_log_y,
                        labels={_x_col: _x_col, _y_col: _y_col},
                    )
                    _sfig.update_traces(
                        marker=dict(size=11, opacity=0.9, line=dict(color="white", width=0.8)),
                    )

                    if len(_xv) >= 2:
                        _slope, _icpt, _rv, _pv, _se = _stats.linregress(_xv, _yv)
                        _xl = np.linspace(_xv.min(), _xv.max(), 200)
                        _yl = _slope * _xl + _icpt
                        if _log_x: _xl = 10 ** _xl
                        if _log_y: _yl = 10 ** _yl
                        _sfig.add_trace(go.Scatter(
                            x=_xl, y=_yl, mode="lines",
                            line=dict(color="#C44E52", width=1.5, dash="dot"),
                            name=f"추세선  r={_rv:+.3f}",
                            hoverinfo="skip",
                        ))
                        _sign = "+" if _icpt >= 0 else "-"
                        _sfig.add_annotation(
                            xref="paper", yref="paper", x=0.01, y=0.99,
                            text=f"y = {_slope:.4f}x {_sign} {abs(_icpt):.4f}",
                            showarrow=False, align="left",
                            bgcolor="rgba(255,255,255,0.85)", bordercolor="#C44E52",
                            borderwidth=1, font=dict(size=12, color="#C44E52"),
                        )

                    _sfig.update_layout(
                        height=520,
                        margin=dict(l=80, r=200, t=40, b=80),
                        plot_bgcolor="white", paper_bgcolor="white",
                        xaxis=dict(
                            title=dict(text=_x_col, font=dict(size=14, color="#222222")),
                            showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                            zeroline=False, showline=True, linecolor="#AAAAAA",
                            tickfont=dict(size=12, color="#222222"),
                        ),
                        yaxis=dict(
                            title=dict(text=_y_col, font=dict(size=14, color="#222222")),
                            showgrid=True, gridcolor="#DDDDDD", griddash="dot",
                            zeroline=False, showline=True, linecolor="#AAAAAA",
                            tickfont=dict(size=12, color="#222222"),
                        ),
                        legend=dict(
                            title=dict(text="업체", font=dict(size=13, color="#222222")),
                            x=1.01, xanchor="left", y=1.0, yanchor="top",
                            bgcolor="rgba(255,255,255,0.9)", bordercolor="#AAAAAA",
                            borderwidth=1, font=dict(size=12, color="#222222"),
                        ),
                        font=dict(family="Arial, sans-serif", color="#222222"),
                    )
                    st.plotly_chart(_sfig, use_container_width=True, key="hvac_corr_scatter", theme=None)

                    if len(_xv) >= 2:
                        _r2 = _rv ** 2
                        _str_k = ("강함" if _r2 >= 0.36 else "보통" if _r2 >= 0.12 else "약함")
                        st.dataframe(pd.DataFrame([{
                            "n": len(_xv),
                            "r": f"{_rv:+.3f}", "R²": f"{_r2:.3f}",
                            "기울기": f"{_slope:.4f}", "절편": f"{_icpt:.4f}",
                            "p-value": f"{_pv:.4e}", "강도": _str_k,
                        }]), hide_index=True, use_container_width=True)

    with tab_anom:
        if not _flag_cols:
            st.info("이상 탐지에 필요한 컬럼이 없습니다.")
        else:
            n_crit  = int((_flags["플래그 수"] >= 2).sum())
            n_watch = int((_flags["플래그 수"] == 1).sum())
            n_ok    = int((_flags["플래그 수"] == 0).sum())

            _am1, _am2, _am3 = st.columns(3)
            _am1.metric("🔴 위험 (2개 이상 플래그)", n_crit)
            _am2.metric("🟠 주의 (1개 플래그)",       n_watch)
            _am3.metric("🟢 정상",                     n_ok)
            st.caption(
                "각 탭에서 이상 업체가 강조 표시됩니다 — "
                "순위: 요금 이상치 callout / 비중: 기본요금 편중 마커 / 면적별 비용 비교: IQR 이상치 경고"
            )
            st.divider()

            # ── Flag heatmap — brands × flag types ───────────────────────────
            _hm_sorted = _flags.sort_values("플래그 수", ascending=False).head(40)
            _hm_brands = [
                ("⛔ " if r["플래그 수"] >= 2 else "⚠ " if r["플래그 수"] == 1 else "") + str(b)[:26]
                for b, r in _hm_sorted.iterrows()
            ]
            _hm_z    = _hm_sorted[_flag_cols].astype(int).values.tolist()
            _hm_text = [["✓" if v else "" for v in row] for row in _hm_z]
            _hm_h    = max(300, len(_hm_brands) * 28 + 80)
            _hm_fig  = go.Figure(go.Heatmap(
                z=_hm_z,
                x=_flag_cols,
                y=_hm_brands,
                colorscale=[[0, "#F0F0F0"], [1, "#C44E52"]],
                showscale=False,
                text=_hm_text,
                texttemplate="%{text}",
                textfont=dict(size=13, color="white"),
                hovertemplate="%{y}<br>%{x}: %{text}<extra></extra>",
                xgap=3, ygap=3,
            ))
            _hm_fig.update_layout(
                height=_hm_h,
                margin=dict(l=10, r=10, t=36, b=60),
                title=dict(text="플래그 매트릭스 — 빨강=이상 감지, 회색=정상", font=dict(size=12)),
                xaxis=dict(side="top", tickfont=dict(size=11), tickangle=-20),
                yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(_hm_fig, use_container_width=True, key="hvac_anom_heatmap", theme=None)

            # ── Flag legend ───────────────────────────────────────────────────
            with st.expander("플래그 기준 안내", expanded=False):
                st.markdown(
                    "| 탭 | 플래그 | 기준 |\n"
                    "|---|---|---|\n"
                    "| 순위 | **요금 이상치** | 소계 요금이 IQR 상한 (Q3 + 1.5×IQR) 초과 |\n"
                    "| 면적별 비용 비교 | **단위면적 이상치** | 원/㎡ 기준 IQR 상한 초과 |\n"
                    "| 면적별 비용 비교 | **단위사용량 이상치** | 원/Mcal 기준 IQR 상한 초과 |\n"
                    "| 비중 | **기본요금 편중** | 기본요금 비중 > 70% |\n"
                    "| — | **🔴 위험** | 2개 이상 플래그 |\n"
                    "| — | **🟠 주의** | 1개 플래그 |"
                )

            # ── Combined flagged brands summary table ─────────────────────────
            _flagged = _flags[_flags["플래그 수"] >= 1].copy()
            _an_ctx_cols = [c for c in [_anom_fee, base_col, usage_fee_col, comm_fee_col, usage_col, area_col] if c and c in brand_agg_all.columns]
            _an_ctx_rename = {
                _anom_fee:     "소계 (원)" if total_col else "전용 합계 (원)",
                base_col:      "기본요금 (원)",
                usage_fee_col: "사용요금 (원)",
                comm_fee_col:  "공용요금 (원)",
                usage_col:     "사용량 (Mcal)",
                area_col:      "면적 (㎡)",
            }
            if _flagged.empty:
                st.success("이상 징후가 감지된 업체가 없습니다.")
            else:
                st.caption(f"이상 징후 업체 {len(_flagged)}개 — 플래그 수 내림차순 (각 탭의 해당 차트에서 상세 확인)")
                _display = _flagged.sort_values(["플래그 수", "등급"], ascending=[False, True])
                _display = _display.join(brand_agg_all[_an_ctx_cols])
                _display = _display.rename(columns={k: v for k, v in _an_ctx_rename.items() if k in _display.columns})
                _display_fmt = _display.copy()
                for _c in _display_fmt.columns:
                    if _c in ["등급", "플래그 수"] + _flag_cols:
                        continue
                    try:
                        _display_fmt[_c] = _display_fmt[_c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")
                    except Exception:
                        pass
                _disp_order = ["등급", "플래그 수"] + _flag_cols + [c for c in _display_fmt.columns if c not in ["등급", "플래그 수"] + _flag_cols]
                st.dataframe(st_safe(_display_fmt[_disp_order]), use_container_width=True, hide_index=False)

    # ── Bottom expanders ───────────────────────────────────────────────────────
    with st.expander("전체 내역", expanded=False):
        group_cols = [c for c in ["유형", "구분"] if c in num_df.columns]
        agg_df = num_df.groupby(group_cols + ["브랜드"])[numeric_cols].sum().reset_index()
        st.dataframe(st_safe(agg_df), use_container_width=True, hide_index=True)

    with st.expander("원본 데이터", expanded=False):
        st.dataframe(st_safe(df), use_container_width=True)


def _mom_tab(curr: pd.DataFrame, prev: pd.DataFrame | None,
             billing_period: str | None = None,
             prev_billing_period: str | None = None,
             mode: str = "mom") -> None:
    """Change comparison tab for 수도광열비 부과 내역.

    mode="mom" → 월별 변화 labels; mode="yoy" → 전년 대비 labels.
    """
    if mode == "yoy":
        _heading = "📅 전년 대비"
        _prev_label = "전년"
        _curr_label = "올해"
        _chart_label = "전년 동월 대비 변화"
        _no_data_msg = "전년 동월 파일에 수도광열비 부과 내역 시트가 없습니다."
        _key_pfx = "billing_yoy"
    else:
        _heading = "📈 월별 변화"
        _prev_label = "전월"
        _curr_label = "이번달"
        _chart_label = "전월 대비 변화"
        _no_data_msg = "이전 달 파일에 수도광열비 부과 내역 시트가 없습니다."
        _key_pfx = "billing_mom"

    period_str = (
        f"{prev_billing_period} → {billing_period}"
        if billing_period and prev_billing_period
        else billing_period or "이번 달"
    )
    st.subheader(f"{_heading}  ({period_str})")

    if prev is None or prev.empty:
        st.info(_no_data_msg)
        return

    # Cost columns to compare
    _COST_COLS = [c for c in [
        "water_total", "elect_total", "hotwater_comm", "heat_total",
        "total_excl", "total_comm", "total",
    ] if c in curr.columns and c in prev.columns]

    _COST_LABELS = {
        "water_total":   "💧 수도 합계",
        "elect_total":   "⚡ 전기 합계",
        "hotwater_comm": "🌡 온수",
        "heat_total":    "🔥 난방 합계",
        "total_excl":    "전용 합계",
        "total_comm":    "공용 합계",
        "total":         "총 합계",
    }

    # Aggregate by brand + building
    id_cols = ["brand", "building"]
    curr_agg = curr.groupby(id_cols)[_COST_COLS].sum().reset_index()
    prev_agg = prev.groupby(id_cols)[_COST_COLS].sum().reset_index()

    # Merge on brand+building; suffix _c = current, _p = previous
    merged = curr_agg.merge(prev_agg, on=id_cols, how="outer", suffixes=("_c", "_p"))
    for c in _COST_COLS:
        merged[f"{c}_c"] = merged[f"{c}_c"].fillna(0)
        merged[f"{c}_p"] = merged[f"{c}_p"].fillna(0)
        merged[f"{c}_chg"] = merged[f"{c}_c"] - merged[f"{c}_p"]
        merged[f"{c}_pct"] = (merged[f"{c}_chg"] / merged[f"{c}_p"].replace(0, float("nan"))) * 100

    # ── KPI row ───────────────────────────────────────────────────────────────
    _kpi_cols = [c for c in ["total", "water_total", "elect_total", "hotwater_comm", "heat_total"]
                 if c in _COST_COLS]
    kc = st.columns(len(_kpi_cols))
    for ci, c in enumerate(_kpi_cols):
        _curr_sum = merged[f"{c}_c"].sum()
        _prev_sum = merged[f"{c}_p"].sum()
        _delta    = _curr_sum - _prev_sum
        _pct      = _delta / _prev_sum * 100 if _prev_sum else 0
        kc[ci].metric(
            _COST_LABELS.get(c, c),
            _fmt_won(_curr_sum * 10000),
            delta=f"{_fmt_won(_delta * 10000, signed=True)} ({_pct:+.1f}%)",
            delta_color="inverse",
        )

    st.divider()

    # ── Column selector + bar chart ───────────────────────────────────────────
    sel_col = st.selectbox(
        "항목",
        _COST_COLS,
        format_func=lambda c: _COST_LABELS.get(c, c),
        key=f"{_key_pfx}_col",
    )
    _chg_col = f"{sel_col}_chg"
    plot_df = merged[["brand", "building", f"{sel_col}_c", f"{sel_col}_p", _chg_col]].copy()
    plot_df = plot_df.sort_values(_chg_col, ascending=True).reset_index(drop=True)

    import plotly.graph_objects as go
    _colors = plot_df[_chg_col].apply(lambda v: "#C44E52" if v > 0 else "#2ca02c").tolist()
    fig = go.Figure(go.Bar(
        x=plot_df[_chg_col],
        y=plot_df["brand"],
        orientation="h",
        marker_color=_colors,
        text=plot_df[_chg_col].apply(lambda v: _fmt_won(v * 10000, signed=True)),
        textposition="outside",
        textfont=dict(size=9, color="#222222"),
        hovertemplate="<b>%{y}</b><br>변화: %{x:+,.1f} 만원<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="#888888", line_width=1)
    fig.update_layout(
        title=f"{_COST_LABELS.get(sel_col, sel_col)} {_chart_label} (만원)",
        height=max(430, len(plot_df) * 22 + 80),
        xaxis_title="변화 (만원)",
        margin=dict(t=55, b=40, l=10, r=130),
        showlegend=False,
        yaxis=dict(tickfont=dict(size=10)),
    )
    _ev = st.plotly_chart(fig, use_container_width=True, key=f"{_key_pfx}_bar_{sel_col}", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = _pts[0].get("y", "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = plot_df[plot_df["brand"] == _brand]
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    st.divider()

    # ── Top / bottom tables ───────────────────────────────────────────────────
    def _fmt_billing_table(df):
        d = df[["brand", "building"]].copy()
        d[_prev_label] = df[f"{sel_col}_p"].apply(lambda v: _fmt_won(v * 10000) if pd.notna(v) else "—")
        d[_curr_label] = df[f"{sel_col}_c"].apply(lambda v: _fmt_won(v * 10000) if pd.notna(v) else "—")
        d["변화"] = df[_chg_col].apply(lambda v: _fmt_won(v * 10000, signed=True) if pd.notna(v) else "—")
        return d

    _n = min(10, len(plot_df))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🔴 증가 상위 {_n}개**")
        st.dataframe(_fmt_billing_table(plot_df.nlargest(_n, _chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)
    with c2:
        st.markdown(f"**🟢 감소 상위 {_n}개**")
        st.dataframe(_fmt_billing_table(plot_df.nsmallest(_n, _chg_col)).reset_index(drop=True), hide_index=True, use_container_width=True)

    st.divider()

    # ── Full change table ─────────────────────────────────────────────────────
    with st.expander("📋 전체 변화 목록", expanded=False):
        _all_cols = ["brand", "building"] + [f"{c}_p" for c in _COST_COLS] + \
                    [f"{c}_c" for c in _COST_COLS] + [f"{c}_chg" for c in _COST_COLS]
        _all_cols = [c for c in _all_cols if c in merged.columns]
        _full_raw = merged[_all_cols].sort_values(f"{_COST_COLS[-1]}_chg", ascending=False).reset_index(drop=True)
        _full_disp = _full_raw[["brand", "building"]].copy()
        for _c in _all_cols:
            if _c in ("brand", "building"):
                continue
            _is_chg = _c.endswith("_chg")
            _full_disp[_c] = _full_raw[_c].apply(
                lambda v, signed=_is_chg: _fmt_won(v * 10000, signed=signed) if pd.notna(v) else "—"
            )
        st.dataframe(_full_disp, hide_index=True, use_container_width=True)


def render_billing_view(
    df: pd.DataFrame,
    prev_df: pd.DataFrame | None = None,
    billing_period: str | None = None,
    prev_billing_period: str | None = None,
    yoy_df: pd.DataFrame | None = None,
    yoy_billing_period: str | None = None,
) -> None:
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

    # Brand search (widget rendered above tabs via brand_search_bar)
    _billing_brand_search = st.session_state.get("billing_brand_search", "").strip().lower()
    if _billing_brand_search:
        fdf = fdf[fdf["brand"].astype(str).str.lower().str.contains(_billing_brand_search, na=False)].copy()

    if fdf.empty:
        st.warning("No data for the selected filters.")
        return

    # ── Report download ────────────────────────────────────────────────────────
    with st.expander("PDF 보고서 다운로드", expanded=False):
        st.caption("현재 필터가 적용된 청구 데이터를 기반으로 업무용 PDF 보고서를 생성합니다.")
        lang_billing = st.radio("언어", ["한국어 (ko)", "English (en)"],
                                horizontal=True, key="billing_report_lang")
        if st.button("PDF 생성", key="billing_gen_report"):
            from billing_report import generate_billing_pdf
            with st.spinner("Generating PDF…"):
                pdf_bytes = generate_billing_pdf(
                    fdf,
                    context={"date": date.today(), "buildings": sel_bldg},
                    lang="ko" if lang_billing.startswith("한") else "en",
                )
            bldg_tag = "all" if "All" in sel_bldg else "_".join(sel_bldg)
            st.download_button(
                label="PDF 다운로드",
                data=pdf_bytes,
                file_name=f"billing_report_{bldg_tag}_{date.today()}.pdf",
                mime="application/pdf",
                key="billing_dl_report",
            )

    # ── Tabs ──
    brand_search_bar("billing")
    tab_mom, tab_yoy, tab_hist, tab_rank, tab_bldg, tab_comp, tab_ratio, tab_perm2 = st.tabs([
        "📈 월별 변화", "📅 전년 대비", "분포", "업체별 순위", "건물별 요약",
        "구성 비율", "공용/전용 비율", "단위면적당",
    ])

    with tab_mom:
        _mom_tab(fdf, prev_df, billing_period, prev_billing_period)

    with tab_yoy:
        _mom_tab(fdf, yoy_df, billing_period, yoy_billing_period, mode="yoy")
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

def _hvac_tab(df: pd.DataFrame) -> None:
    needed = {"hvac_excl", "hvac_comm", "building", "brand"}
    if not needed.issubset(df.columns):
        st.info("냉난방 컬럼을 찾을 수 없습니다.")
        return

    df = df.copy()
    df["hvac_total"] = df["hvac_excl"] + df["hvac_comm"]

    # Summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("냉난방 전용 합계", f"{df['hvac_excl'].sum():,.2f} 만원")
    c2.metric("냉난방 공용 합계", f"{df['hvac_comm'].sum():,.2f} 만원")
    c3.metric("냉난방 합계",      f"{df['hvac_total'].sum():,.2f} 만원")

    # Building-level grouped bar
    bldg = (df.groupby("building")[["hvac_excl", "hvac_comm"]]
              .sum().reset_index()
              .sort_values("hvac_excl", ascending=False))
    fig_bldg = go.Figure([
        go.Bar(name="전용", x=bldg["building"], y=bldg["hvac_excl"],
               marker_color=_HEAT_COLOR,
               text=[f"{v:,.2f}" for v in bldg["hvac_excl"]], textposition="outside",
               hovertemplate="<b>%{x}</b> 전용: %{y:,.2f} 만원<extra></extra>"),
        go.Bar(name="공용", x=bldg["building"], y=bldg["hvac_comm"],
               marker_color="#E08080",
               text=[f"{v:,.2f}" for v in bldg["hvac_comm"]], textposition="outside",
               hovertemplate="<b>%{x}</b> 공용: %{y:,.2f} 만원<extra></extra>"),
    ])
    fig_bldg.update_layout(
        **_BASE_LAYOUT,
        barmode="group",
        title=dict(text="<b>건물별 냉난방 비용 (만원)</b>", font=dict(size=14), x=0),
        height=380,
        xaxis=dict(title="건물", showgrid=False),
        yaxis=dict(title="만원", showgrid=True, gridcolor=_GRID, rangemode="tozero"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08),
        margin=dict(l=60, r=20, t=80, b=50),
    )
    st.plotly_chart(fig_bldg, use_container_width=True, key="billing_hvac_bldg_chart")

    # Top brands by hvac_excl
    top_n = st.slider("상위 브랜드 수", 5, min(50, len(df)), 20, key="billing_hvac_topn")
    brand_grp = (df.groupby("brand")["hvac_excl"]
                   .sum().reset_index()
                   .sort_values("hvac_excl", ascending=False)
                   .head(top_n))
    fig_brand = go.Figure(go.Bar(
        x=brand_grp["brand"], y=brand_grp["hvac_excl"],
        marker_color=_HEAT_COLOR,
        text=[f"{v:,.2f}" for v in brand_grp["hvac_excl"]], textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b>: %{y:,.2f} 만원<extra></extra>",
    ))
    fig_brand.update_layout(
        **_BASE_LAYOUT,
        title=dict(text=f"<b>상호별 냉난방 전용 Top {top_n} (만원)</b>", font=dict(size=14), x=0),
        height=400,
        xaxis=dict(tickangle=-40, tickfont=dict(size=9), showgrid=False),
        yaxis=dict(title="만원", showgrid=True, gridcolor=_GRID, rangemode="tozero"),
        margin=dict(l=60, r=20, t=70, b=100),
        showlegend=False,
    )
    st.plotly_chart(fig_brand, use_container_width=True, key="billing_hvac_brand_chart")

    # Building summary table
    tbl = (df.groupby("building")
             .agg(전용=("hvac_excl", "sum"), 공용=("hvac_comm", "sum"),
                  합계=("hvac_total", "sum"), 건수=("brand", "count"))
             .reset_index().rename(columns={"building": "건물"}))
    tbl[["전용", "공용", "합계"]] = tbl[["전용", "공용", "합계"]].round(2)
    st.dataframe(tbl, hide_index=True, use_container_width=True)

def _ranking_tab(df: pd.DataFrame) -> None:
    _n = len(df)
    if _n >= 2:
        top_n = st.slider("표시 업체 수", min(10, _n - 1), _n, min(30, _n), key="rank_n")
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
        "정렬 기준", ["현재 뷰", "합계"],
        horizontal=True, key="rank_sort",
    )
    if sort_key == "합계" and _ref_col and _ref_col in df.columns:
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
    st.markdown(f"**{len(out)}개 업체** — {label} 내림차순")
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
        "표시 범위", ["전체", "상위", "중간", "하위"],
        horizontal=True, key="hist_show",
    )
    label = f"{tail_pct}%"

    if show_mode == "전체":
        tbl = df[display_cols].dropna(subset=[val_col]).sort_values(val_col, ascending=False).copy()
        st.markdown(f"**전체 {len(tbl)}개 업체** — 내림차순")
    elif show_mode == "상위":
        tbl = df[df[val_col] >= hi][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**상위 {label} (≥ {hi:,.2f})** — {len(tbl)}개 업체")
    elif show_mode == "중간":
        tbl = df[(df[val_col] > lo) & (df[val_col] < hi)][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**중간** ({lo:,.2f} – {hi:,.2f}) — {len(tbl)}개 업체")
    else:  # 하위
        tbl = df[df[val_col] <= lo][display_cols].sort_values(val_col, ascending=False).copy()
        st.markdown(f"**하위 {label} (≤ {lo:,.2f})** — {len(tbl)}개 업체")

    tbl = add_display_index(tbl)
    st.dataframe(st_safe(tbl), hide_index=True, use_container_width=True,
                 height=min(35 * len(tbl) + 38, 700))
    download_df_as_excel(tbl, filename=f"billing_hist_{sel_util}_{view_mode}_{show_mode}.xlsx", sheet_name="hist")


def _building_tab(df: pd.DataFrame) -> None:

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

    st.markdown("**건물별 합계**")
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
    present = [(c, lbl, clr) for c, lbl, clr in _COMP_COLS if c in df.columns]
    if not present:
        st.warning("No utility cost columns found.")
        return

    seg_cols = [c for c, _, _ in present]

    mode = st.radio("차트 형식", ["비율 (100% 누적)", "금액 (만원)"],
                    horizontal=True, key="comp_mode")

    _n = len(df)
    top_n = st.slider("표시 업체 수", min(10, _n), _n, min(40, _n), key="comp_n") if _n > 1 else _n

    sort_col = next((c for c in ["total"] + seg_cols if c in df.columns), seg_cols[0])
    plot_df = df.sort_values(sort_col, ascending=False).head(top_n).iloc[::-1].copy()

    if mode.startswith("비율"):
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
    st.markdown(f"**{len(out)}개 업체** — 총 요금 내림차순")
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
    st.caption("공용 요금 비율이 비정상적으로 높은 업체를 강조합니다.")

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
    top_n = st.slider("표시 업체 수", min(10, _n), _n, min(40, _n), key="ratio_n") if _n > 1 else _n
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
    c1.metric("분석 업체 수", len(wdf.dropna(subset=["comm_ratio"])))
    c2.metric("공용 비율 중앙값", f"{median_ratio:.1f}%")
    c3.metric("이상치 기준", f"{upper_fence:.1f}%")
    c4.metric("이상치 업체 수", n_outliers)

    # Table: all brands with flag
    tbl = wdf.sort_values("comm_ratio", ascending=False).copy()
    tbl["outlier"] = tbl["comm_ratio"] >= upper_fence
    show_cols = ["brand", "building", "floor", comm, excl, "comm_ratio", "outlier"]
    if tot_col in tbl.columns:
        show_cols.insert(4, tot_col)
    show_cols = [c for c in show_cols if c in tbl.columns]
    out = add_display_index(tbl[show_cols].copy())
    st.markdown(f"**{n_outliers}개 이상치 업체** (공용 비율 ≥ {upper_fence:.1f}%)")
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
    st.caption("단위: 만원/m² · 면적 대비 공정 비교를 위해 입주사 면적으로 정규화한 값입니다.")

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
    top_n = st.slider("표시 업체 수", min(10, _n), _n, min(40, _n), key="perm2_n") if _n > 1 else _n

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
    c1.metric("면적 있는 업체", len(vals))
    c2.metric("중앙값 (만원/m²)", f"{median_pm2:.4f}")
    c3.metric("평균 (만원/m²)",   f"{mean_pm2:.4f}")
    c4.metric("고가 이상치",      n_high)
    c5.metric("저가 이상치",      n_low)

    # Full ranked table with per-m² column + flag
    tbl = wdf.sort_values(pm2_col, ascending=False).copy()
    tbl["high_outlier"] = tbl[pm2_col] >= upper_fence
    tbl["low_outlier"]  = (lower_fence > 0) & (tbl[pm2_col] <= lower_fence)
    show_cols = ["brand", "building", "floor", "size_m2", col, pm2_col,
                 "high_outlier", "low_outlier"]
    show_cols = [c for c in show_cols if c in tbl.columns]
    out = add_display_index(tbl[show_cols].copy())
    st.markdown(f"**{len(out)}개 업체** (면적 있음) — {lbl}/m² 내림차순")
    st.dataframe(st_safe(out), hide_index=True, use_container_width=True,
                 height=min(35 * len(out) + 38, 700))
    download_df_as_excel(out, filename=f"billing_per_m2_{lbl}.xlsx", sheet_name="per_m2")
