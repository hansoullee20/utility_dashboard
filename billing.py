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

def render_hvac_view(df: pd.DataFrame) -> None:
    st.subheader("관리비 고지서 EHP 열(냉난방)")
    st.caption("▣ 열(냉난방)사용 내역")

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
    amount_col    = next((c for c in df.columns if "전용" in c and "면적" not in c and "부과" not in c and "요금" not in c), None)
    total_col     = next((c for c in df.columns if "소계" in c), None)
    numeric_cols  = [c for c in [usage_col, base_col, usage_fee_col, comm_fee_col, amount_col, total_col] if c]

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
        st.markdown(f"#### FCU 요금 정합성 검증: {_val_label} = {_formula}")
        st.caption(
            f"{_val_label}는 {_formula}의 합으로 구성됩니다 (Excel: P = K + M + O). "
            "두 값이 일치하면 요금 산정 로직이 정확히 반영된 것이며, "
            "불일치가 있을 경우 데이터 입력 오류 또는 별도 조정분(할인·가산금 등)이 포함된 것으로 볼 수 있습니다."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("검증 대상", f"{n_total}건")
        c2.metric("일치", f"{n_match}건", f"{match_rate:.1f}%")
        c3.metric("불일치", f"{n_mismatch}건")
        c4.metric("최대 차이", f"{abs_diff.max():,.0f}원" if n_total else "—")

        if n_mismatch == 0:
            st.success(f"전체 {n_total}건 모두 일치 — 요금 데이터 정합성이 확인되었습니다.")
        else:
            st.warning(
                f"{n_mismatch}건에서 {_val_label}와 {_formula}이 다릅니다. "
                f"평균 차이 {abs_diff[mismatch_mask].mean():,.0f}원, 최대 차이 {abs_diff.max():,.0f}원. "
                "할인·연체·기타 조정분이 포함되어 있을 가능성이 있습니다."
            )
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
            st.markdown(
                "- **업체 수**: 해당 항목에 값이 있는(0 제외) 업체 수\n"
                "- **합계**: 표시된 업체들의 값을 모두 더한 총액\n"
                "- **평균**: 합계 ÷ 업체 수\n"
                "- **중앙값**: 업체들을 금액 순으로 정렬했을 때 정중앙에 위치한 값으로, 극단값의 영향을 받지 않음\n"
                "- **표준편차**: 업체 간 편차가 클수록 높으며, 요금 분포의 불균형 정도를 나타냄\n"
                "- **최대 / 최소**: 가장 높은 / 낮은 업체의 값"
                + (f"\n- **합계 비중**: 해당 항목의 합계가 전용 합계 총액({total_sum:,.0f}원) 중 차지하는 비율" if total_sum else "")
            )

        # ── Statistical interpretation ─────────────────────────────────────────
        insights = []
        for label, s in rows:
            s = s.replace(0, pd.NA).dropna()
            if s.empty or len(s) < 2:
                continue
            mean, median = s.mean(), s.median()
            std, mx, mn  = s.std(), s.max(), s.min()
            cv           = std / mean if mean else 0
            skew_ratio   = mean / median if median else 1

            lines = [f"**{label}**"]
            # Distribution shape
            if skew_ratio > 1.5:
                lines.append(f"평균({mean:,.0f})이 중앙값({median:,.0f})보다 {skew_ratio:.1f}배 높아 일부 고액 업체가 평균을 끌어올리는 우편향 분포입니다.")
            elif skew_ratio < 0.67:
                lines.append(f"평균({mean:,.0f})이 중앙값({median:,.0f})보다 낮아 소수의 저액 업체가 평균을 낮추는 좌편향 분포입니다.")
            else:
                lines.append(f"평균({mean:,.0f})과 중앙값({median:,.0f})이 유사하여 비교적 고른 분포를 보입니다.")
            # Variability
            if cv > 1.0:
                lines.append(f"변동계수(CV={cv:.2f})가 1을 초과하여 업체 간 요금 편차가 매우 큽니다.")
            elif cv > 0.5:
                lines.append(f"변동계수(CV={cv:.2f})로 업체 간 요금 편차가 다소 있습니다.")
            else:
                lines.append(f"변동계수(CV={cv:.2f})로 업체 간 요금이 비교적 균일합니다.")
            # Max/min spread
            if mn > 0:
                lines.append(f"최대({mx:,.0f})는 최소({mn:,.0f})의 {mx/mn:.0f}배로, 업체 간 요금 규모 차이가 {'매우 큽니다' if mx/mn > 10 else '있습니다'}.")
            # Total share context
            if total_sum and total_sum > 0:
                share = s.sum() / total_sum * 100
                lines.append(f"전용 합계 총액의 {share:.1f}%를 차지합니다.")
            insights.append("\n  ".join(lines))

        if insights:
            with st.expander("통계 해석"):
                for insight in insights:
                    st.markdown(insight)
                    st.divider()

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
        fig = go.Figure(go.Histogram(
            x=clean,
            marker_color=color,
            marker_line=dict(color="white", width=1),
            opacity=0.85,
        ))
        fig.update_layout(
            height=300,
            margin=dict(t=20, b=60, l=60, r=20),
            xaxis=_axis(title=xlab),
            yaxis=_axis(grid=True, title="업체 수"),
            bargap=0.05,
            **_LAYOUT_BASE,
        )
        st.plotly_chart(fig, use_container_width=True, key=key, theme=None)

    # ── Overall statistical analysis ──────────────────────────────────────────
    st.markdown("#### 전체 통계 분석")
    brand_agg_all = num_df.groupby("브랜드")[numeric_cols].sum()

    # Top-line metrics
    _mc = st.columns(4)
    _mc[0].metric("총 브랜드 수", f"{num_df['브랜드'].nunique():,}")
    if usage_col:
        _mc[1].metric("총 사용량 (Mcal)", f"{brand_agg_all[usage_col].sum():,.0f}")
    if amount_col:
        _mc[2].metric("총 전용 합계 (원)", f"{brand_agg_all[amount_col].sum():,.0f}")
    if base_col and usage_fee_col:
        _base_ratio = brand_agg_all[base_col].sum() / brand_agg_all[amount_col].sum() * 100 if amount_col and brand_agg_all[amount_col].sum() else 0
        _mc[3].metric("기본요금 비중", f"{_base_ratio:.1f}%")

    # Per-column stats across all brands
    _stat_rows = []
    for col, label in [(c, l) for c, l in [
        (usage_col,     "사용량 (Mcal)"),
        (base_col,      "기본요금 (원)"),
        (usage_fee_col, "사용요금 (원)"),
        (comm_fee_col,  "공용요금 (원)"),
        (amount_col,    "전용 합계 (원)"),
        (total_col,     "소계 (원)"),
    ] if c]:
        s = brand_agg_all[col].replace(0, pd.NA).dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        _stat_rows.append({
            "항목":    label,
            "업체 수": len(s),
            "합계":    f"{s.sum():,.0f}",
            "평균":    f"{s.mean():,.0f}",
            "중앙값":  f"{s.median():,.0f}",
            "표준편차": f"{s.std():,.0f}",
            "Q1 (25%)": f"{q1:,.0f}",
            "Q3 (75%)": f"{q3:,.0f}",
            "IQR":     f"{(q3 - q1):,.0f}",
            "최대":    f"{s.max():,.0f}",
            "최소":    f"{s.min():,.0f}",
        })
    if _stat_rows:
        st.dataframe(pd.DataFrame(_stat_rows), use_container_width=True, hide_index=True)

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


    # Missing data summary
    _total_brands = num_df["브랜드"].nunique()
    _missing_rows = []
    for col, label in [(c, l) for c, l in [
        (usage_col,     "사용량 (Mcal)"),
        (base_col,      "기본요금 (원)"),
        (usage_fee_col, "사용요금 (원)"),
        (amount_col,    "전용 합계 (원)"),
    ] if c]:
        _agg = brand_agg_all[col]
        n_zero    = int((_agg == 0).sum())
        n_missing = n_zero  # NaN already replaced with 0 during load
        _missing_rows.append({
            "항목":              label,
            "전체 업체":         _total_brands,
            "데이터 있음":       _total_brands - n_missing,
            "0 (누락 또는 미부과)": n_missing,
            "비율":              f"{n_missing / _total_brands * 100:.1f}%" if _total_brands else "—",
        })
    if _missing_rows:
        with st.expander("결측 데이터 요약", expanded=False):
            st.dataframe(pd.DataFrame(_missing_rows), use_container_width=True, hide_index=True)

            st.caption(
                "모든 빈 값(NaN)은 로딩 시 0으로 처리됩니다. "
                "따라서 0으로 표시된 업체는 실제로 해당 항목이 없어 0원으로 부과된 경우일 수도 있고, "
                "원본 데이터에 값이 입력되지 않아 누락된 경우일 수도 있습니다. "
                "두 경우를 구분하려면 원본 Excel을 직접 확인하세요."
            )

            # Per-brand missing pattern (which columns are missing per brand)
            _pattern_cols = [(c, l) for c, l in [
                (usage_col, "사용량"), (base_col, "기본요금"),
                (usage_fee_col, "사용요금"), (amount_col, "전용 합계"),
            ] if c]
            if len(_pattern_cols) > 1:
                _pat_df = brand_agg_all[[c for c, _ in _pattern_cols]].copy()
                _pat_df.columns = [l for _, l in _pattern_cols]
                _missing_pattern = _pat_df[(_pat_df == 0).any(axis=1) | _pat_df.isna().any(axis=1)].copy()
                _missing_pattern = _missing_pattern.applymap(lambda v: "—" if (pd.isna(v) or v == 0) else "✔")
                if not _missing_pattern.empty:
                    st.caption("항목별 누락이 있는 업체")
                    st.dataframe(_missing_pattern, use_container_width=True)

    # Concentration: top-5 brands share
    if amount_col:
        _total = brand_agg_all[amount_col].sum()
        _top5  = brand_agg_all[amount_col].nlargest(5)
        _top5_share = _top5.sum() / _total * 100 if _total else 0
        with st.expander("상위 5개 업체 집중도"):
            st.caption(
                f"집중도(concentration ratio)는 상위 5개 업체의 전용 합계를 합산한 뒤 전체 총액으로 나눈 값으로, "
                f"요금이 소수 업체에 얼마나 몰려 있는지를 나타냅니다. "
                f"현재 상위 5개 업체의 합계는 {_top5.sum():,.0f}원이고 전체 총액은 {_total:,.0f}원으로, 집중도는 {_top5_share:.1f}%입니다.\n\n"
                f"아래 표의 비중 열은 각 업체의 전용 합계가 전체 총액({_total:,.0f}원) 중 몇 퍼센트인지를 나타내며, "
                "업체별 전용 합계 ÷ 전체 총액 × 100으로 계산됩니다."
            )
            st.latex(
                r"\text{집중도} = \frac{\sum_{i=1}^{5} \text{전용합계}_i}{\sum_{\text{전체}} \text{전용합계}} \times 100"
                rf"= \frac{{{_top5.sum():,.0f}}}{{{_total:,.0f}}} \times 100 = {_top5_share:.1f}\%"
            )
            _top5_df = _top5.reset_index()
            _top5_df.columns = ["브랜드", "전용 합계 (원)"]
            _top5_df["비중"] = (_top5_df["전용 합계 (원)"] / _total * 100).map(lambda x: f"{x:.1f}%")
            _top5_df["전용 합계 (원)"] = _top5_df["전용 합계 (원)"].map(lambda x: f"{x:,.0f}")
            st.dataframe(_top5_df, use_container_width=True, hide_index=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_rank, tab_prop, tab_summary = st.tabs(["업체별 요금·사용량", "기본·사용요금 비중", "전체 내역"])

    # ── Brand data summary (computed once, used across tabs) ──────────────────
    ref_col = amount_col or usage_col
    brands_with_data = num_df.groupby("브랜드")[ref_col].sum().dropna() if ref_col else pd.Series(dtype=float)
    brands_with_data = brands_with_data[brands_with_data > 0]
    valid_brands = set(brands_with_data.index)
    fdf = num_df[num_df["브랜드"].isin(valid_brands)]
    total_brands = num_df["브랜드"].nunique()
    no_data_brands = num_df[~num_df["브랜드"].isin(valid_brands)]["브랜드"].dropna().unique().tolist()

    with tab_rank:
        # ── Combined summary ──────────────────────────────────────────────────
        _sum_parts = []
        if amount_col and usage_col:
            _top1_fee   = brand_agg_all[amount_col].idxmax()
            _top1_usage = brand_agg_all[usage_col].idxmax()
            _fee_total  = brand_agg_all[amount_col].sum()
            _usage_total = brand_agg_all[usage_col].sum()
            _top1_fee_share   = brand_agg_all.loc[_top1_fee,   amount_col] / _fee_total   * 100
            _top1_usage_share = brand_agg_all.loc[_top1_usage, usage_col]  / _usage_total * 100
            _sum_parts.append(
                f"전용 합계 최다 업체는 **{_top1_fee}** ({_top1_fee_share:.1f}% 차지), "
                f"사용량 최다 업체는 **{_top1_usage}** ({_top1_usage_share:.1f}% 차지)."
            )
            _fee_cv = brand_agg_all[amount_col].std() / brand_agg_all[amount_col].mean()
            _sum_parts.append(
                f"요금 분포 변동계수 {_fee_cv:.2f} — " +
                ("업체 간 요금 격차가 매우 크며 소수 고액 업체가 존재합니다." if _fee_cv > 1
                 else "업체 간 요금이 비교적 균등합니다.")
            )
        if len(_sum_parts):
            st.info("  \n".join(_sum_parts))

        fee_check_cols = {k: v for k, v in [
            ("기본요금", base_col), ("사용요금", usage_fee_col), ("전용 합계", amount_col)
        ] if v}
        brand_fee_agg = num_df.groupby("브랜드")[list(fee_check_cols.values())].sum()
        n_total = num_df["브랜드"].nunique()
        coverage = pd.DataFrame([{
            "항목": label,
            "데이터 있는 업체": int((brand_fee_agg[col] > 0).sum()),
            "데이터 없는 업체": n_total - int((brand_fee_agg[col] > 0).sum()),
        } for label, col in fee_check_cols.items()])
        st.dataframe(coverage, use_container_width=True, hide_index=True)

        top_n = st.slider("표시 업체 수", min_value=1, max_value=max(1, len(brands_with_data)),
                          value=min(20, len(brands_with_data)), step=1, key="hvac_top_n")
        sub_fee, sub_usage = st.tabs(["요금", "사용량"])

        with sub_usage:
            if usage_col:
                grp = fdf.groupby("브랜드")[usage_col].sum().dropna()
                grp = grp[grp > 0].sort_values(ascending=False).head(top_n)
                _stats_table([("사용량 (Mcal)", grp)])
                _hbar(grp, "사용량 (Mcal)", _WATER_COLOR, "hvac_rank_usage")

        with sub_fee:
            if amount_col:
                # Radio: show 소계 as the primary combined view when available
                _fee_radio_opts = (["소계"] if total_col else ["전용 합계"]) + \
                                  (["기본요금"] if base_col else []) + \
                                  (["사용요금"] if usage_fee_col else []) + \
                                  (["공용요금"] if comm_fee_col else [])
                fee_sel = st.radio(
                    "요금 항목",
                    _fee_radio_opts,
                    horizontal=True, key="hvac_fee_sel",
                )
                # Anchor business list to 전용 합계 for all graphs
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
                    _stacked_traces = [("기본요금", "#9B59B6"), ("사용요금", "#C084D4")]
                    if comm_fee_col:
                        _stacked_traces.append(("공용요금", "#7D3C98"))
                    for label, color in _stacked_traces:
                        if label in pivot.columns:
                            fig.add_trace(go.Bar(
                                name=label, x=pivot[label], y=pivot.index.astype(str),
                                orientation="h", marker_color=color,
                            ))
                    totals = pivot.sum(axis=1)
                    annotations = [
                        dict(
                            x=totals[brand], y=brand,
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
                    _stat_entries = [("기본요금 (원)", pivot["기본요금"]), ("사용요금 (원)", pivot["사용요금"])]
                    if comm_fee_col:
                        _stat_entries.append(("공용요금 (원)", pivot["공용요금"]))
                    _denom_lbl = "소계 (원)" if total_col else "전용 합계 (원)"
                    _stat_entries.append((_denom_lbl, totals))
                    _total_fee = totals.sum()
                    _stats_table(_stat_entries, total_sum=_total_fee if _total_fee else None)
                    st.plotly_chart(fig, use_container_width=True, key="hvac_fee_stacked", theme=None)
                else:
                    if fee_sel == "기본요금":
                        fee_col, color = base_col, "#9B59B6"
                    elif fee_sel == "사용요금":
                        fee_col, color = usage_fee_col, "#C084D4"
                    else:  # 공용요금
                        fee_col, color = comm_fee_col, "#7D3C98"
                    grp = sub_fdf.groupby("브랜드")[fee_col].sum().reindex(top_brands).fillna(0)
                    grp = grp.sort_values(ascending=False)
                    _overall_total = sub_fdf.groupby("브랜드")[amount_col].sum().reindex(top_brands).fillna(0).sum()
                    _stats_table([(f"{fee_sel} (원)", grp)], total_sum=_overall_total if _overall_total else None)
                    _hbar(grp, f"{fee_sel} (원)", color, "hvac_fee_single")

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
            _ratio_pct_all   = _ratio * 100
            _denom_label     = "소계" if total_col else "전용 합계"

            # ── Combined summary ──────────────────────────────────────────────
            _dominant_overall = max(
                [("기본요금", _total_base_pct), ("사용요금", _total_usage_pct), ("공용요금", _total_comm_pct)],
                key=lambda x: x[1]
            )[0]
            _n_base_dom  = int((_ratio_pct_all > 110).sum())
            _n_usage_dom = int((_ratio_pct_all < 90).sum())
            _n_equal     = len(_ratio_pct_all) - _n_base_dom - _n_usage_dom
            _high_base_n = int((_base_pct > 80).sum())
            _comm_line   = f" / 공용 {_total_comm_pct:.1f}%" if comm_fee_col else ""
            _prop_summary = (
                f"소계 구성 (Excel P = K + M + O): "
                f"기본 {_total_base_pct:.1f}% / 사용 {_total_usage_pct:.1f}%{_comm_line}.  \n"
                f"**{_dominant_overall}**이 가장 큰 비중을 차지합니다.  \n"
                f"업체별 기본:사용 비율 — 기본요금 우세 {_n_base_dom}개, 사용요금 우세 {_n_usage_dom}개, 동일 {_n_equal}개."
            )
            if _high_base_n:
                _prop_summary += f"  \n기본요금 비중 80% 초과 업체 {_high_base_n}개 — 고정 부담이 두드러집니다."
            st.info(_prop_summary)

            _sub_pct, _sub_ratio = st.tabs(["비중", "비율"])

            with _sub_pct:
                _n_stat_cols = 4 if not comm_fee_col else 4
                for _fee_lbl, _fee_s in [("기본요금", _base_pct), ("사용요금", _usage_pct)] + \
                                        ([("공용요금", _comm_pct)] if comm_fee_col else []):
                    _sc = st.columns(4)
                    _sc[0].metric(f"{_fee_lbl} 비중 평균",   f"{_fee_s.mean():.1f}%")
                    _sc[1].metric(f"{_fee_lbl} 비중 중앙값", f"{_fee_s.median():.1f}%")
                    _sc[2].metric(f"{_fee_lbl} 비중 최대",   f"{_fee_s.max():.1f}%")
                    _sc[3].metric(f"{_fee_lbl} 비중 최소",   f"{_fee_s.min():.1f}%")

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

                with st.expander("비중 해석"):
                    _base_std = _base_pct.std()
                    _comp_str = f"기본요금 {_total_base_pct:.1f}%, 사용요금 {_total_usage_pct:.1f}%" + \
                                (f", 공용요금 {_total_comm_pct:.1f}%" if comm_fee_col else "")
                    st.markdown(
                        f"- {_denom_label} 구성 (P = K + M{' + O' if comm_fee_col else ''}): **{_comp_str}**. "
                        + ("기본요금이 가장 높아 사용량과 무관한 고정 부담이 큰 구조입니다."
                           if _total_base_pct == max(_total_base_pct, _total_usage_pct, _total_comm_pct)
                           else "사용요금이 가장 높아 실제 냉난방 사용량이 요금의 주요 결정 요인입니다.")
                    )
                    st.markdown(
                        f"- 업체별 기본요금 비중의 표준편차가 {_base_std:.1f}%p로 "
                        + (f"크며, 최소 {_base_pct.min():.1f}%에서 최대 {_base_pct.max():.1f}%까지 분포합니다. 업체마다 요금 구성 비율이 크게 다름을 의미합니다."
                           if _base_std > 20 else
                           "작아 대부분의 업체가 유사한 요금 구성 비율을 보입니다.")
                    )

                    _ref_cols_for_table = [c for c in [base_col, usage_fee_col, comm_fee_col, _denom_col] if c]
                    _ref_rename_map = {
                        base_col:      "기본요금 (원)",
                        usage_fee_col: "사용요금 (원)",
                        _denom_col:    "소계 (원)" if total_col else "전용 합계 (원)",
                    }
                    if comm_fee_col:
                        _ref_rename_map[comm_fee_col] = "공용요금 (원)"

                    _high_base = _agg[_agg["기본요금 비중(%)"] > 80][_ref_cols_for_table + ["기본요금 비중(%)"]].copy()
                    if not _high_base.empty:
                        st.markdown(f"- 기본요금 비중이 **80% 초과**인 업체 {len(_high_base)}개 — 사용량이 거의 없음에도 기본요금만 부과되는 업체입니다.")
                        _high_base = _high_base.rename(columns=_ref_rename_map)
                        _num_cols_hb = [c for c in _high_base.columns if c != "기본요금 비중(%)"]
                        for _c in _num_cols_hb:
                            _high_base[_c] = _high_base[_c].map(lambda v: f"{v:,.0f}")
                        _high_base["기본요금 비중(%)"] = _high_base["기본요금 비중(%)"].map(lambda v: f"{v:.1f}%")
                        st.dataframe(st_safe(_high_base.sort_values("기본요금 비중(%)", ascending=False)), use_container_width=True)

                    _no_base_cols = [c for c in [base_col, usage_fee_col, comm_fee_col, _denom_col] if c]
                    _no_base = _agg[_agg[base_col] == 0][_no_base_cols].copy()
                    if not _no_base.empty:
                        st.markdown(f"- **기본요금이 0**인 업체 {len(_no_base)}개 — 사용요금만 부과되고 있습니다.")
                        _no_base = _no_base.rename(columns=_ref_rename_map)
                        for _c in _no_base.columns:
                            _no_base[_c] = _no_base[_c].map(lambda v: f"{v:,.0f}")
                        st.dataframe(st_safe(_no_base), use_container_width=True)

                _col_donut, _col_bar = st.columns([1, 2])
                with _col_donut:
                    _donut_labels = ["기본요금", "사용요금"]
                    _donut_values = [_total_base_pct, _total_usage_pct]
                    _donut_colors = ["#9B59B6", "#C084D4"]
                    if comm_fee_col and _total_comm_pct > 0:
                        _donut_labels.append("공용요금")
                        _donut_values.append(_total_comm_pct)
                        _donut_colors.append("#7D3C98")
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
                        height=280,
                        margin=dict(t=30, b=10, l=10, r=10),
                        showlegend=False,
                        annotations=[dict(text="전체", x=0.5, y=0.5,
                                          font=dict(size=13, color="#000000"), showarrow=False)],
                        **_LAYOUT_BASE,
                    )
                    st.plotly_chart(_fig_donut, use_container_width=True, key="hvac_donut", theme=None)

                with _col_bar:
                    _sorted = _agg.sort_values("기본요금 비중(%)", ascending=True)
                    _fig_prop = go.Figure()
                    _bar_traces = [("기본요금 비중(%)", "#9B59B6"), ("사용요금 비중(%)", "#C084D4")]
                    if comm_fee_col:
                        _bar_traces.append(("공용요금 비중(%)", "#7D3C98"))
                    for _lbl, _clr in _bar_traces:
                        _fig_prop.add_trace(go.Bar(
                            name=_lbl.replace(" 비중(%)", ""),
                            x=_sorted[_lbl],
                            y=_sorted.index.astype(str),
                            orientation="h",
                            marker_color=_clr,
                            text=_sorted[_lbl].map(lambda v: f"{v:.0f}%"),
                            textposition="inside",
                            textfont=dict(color="white", size=10),
                        ))
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

            with _sub_ratio:
                _ratio_pct = _ratio * 100  # express as percentage
                _rc = st.columns(4)
                _rc[0].metric("평균 비율",   f"{_ratio_pct.mean():.1f}%")
                _rc[1].metric("중앙값 비율", f"{_ratio_pct.median():.1f}%")
                _rc[2].metric("최대",        f"{_ratio_pct.max():.1f}%")
                _rc[3].metric("최소",        f"{_ratio_pct.min():.1f}%")

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

                st.dataframe(
                    _ratio_table.style.applymap(
                        lambda v: _label_colors.get(v, ""),
                        subset=["우세 항목"],
                    ),
                    use_container_width=True,
                )

                with st.expander("비율 해석"):
                    _above1_mask = _ratio_pct > 100
                    _below1_mask = _ratio_pct < 100
                    _above1 = int(_above1_mask.sum())
                    _below1 = int(_below1_mask.sum())

                    st.markdown("- 비율 = 기본요금 ÷ 사용요금 × 100(%)으로, 100%이면 두 항목이 동일, 100% 초과이면 기본요금이 더 높고 100% 미만이면 사용요금이 더 높음을 의미합니다.")
                    st.markdown(
                        f"- 평균 {_ratio_pct.mean():.1f}% — " + (
                            "전반적으로 기본요금이 사용요금보다 높아 고정 부담이 큰 구조입니다."
                            if _ratio_pct.mean() > 100 else
                            "전반적으로 사용요금이 기본요금보다 높아 실제 사용량이 요금을 주도합니다."
                        )
                    )

                    def _ratio_subtable(mask, label):
                        if not mask.any():
                            return
                        st.markdown(f"- **{label}** ({int(mask.sum())}개)")
                        _t = _agg.loc[mask.index[mask], [base_col, usage_fee_col]].copy()
                        _t.columns = ["기본요금 (원)", "사용요금 (원)"]
                        _t["비율 (%)"] = _ratio_pct[mask].map(lambda v: f"{v:.1f}%")
                        for _c in ["기본요금 (원)", "사용요금 (원)"]:
                            _t[_c] = _t[_c].map(lambda v: f"{v:,.0f}")
                        st.dataframe(st_safe(_t), use_container_width=True)

                    _ratio_subtable(_above1_mask, "기본요금 > 사용요금")
                    _ratio_subtable(_below1_mask, "기본요금 < 사용요금")
                    _equal_mask = ~_above1_mask & ~_below1_mask
                    _ratio_subtable(_equal_mask, "기본요금 ≈ 사용요금 (동일)")

                _ratio_sorted = _ratio_pct.sort_values(ascending=False)
                _fig_ratio = go.Figure(go.Bar(
                    x=_ratio_sorted.values,
                    y=_ratio_sorted.index.astype(str),
                    orientation="h",
                    marker_color=["#9B59B6" if v >= 100 else "#C084D4" for v in _ratio_sorted.values],
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
        else:
            st.info("기본요금 또는 사용요금 컬럼이 없어 비중 분석을 표시할 수 없습니다.")

    with tab_summary:
        group_cols = [c for c in ["유형", "구분"] if c in num_df.columns]
        agg_df = num_df.groupby(group_cols + ["브랜드"])[numeric_cols].sum().reset_index()
        st.dataframe(st_safe(agg_df), use_container_width=True, hide_index=True)

    with st.expander("원본 데이터", expanded=False):
        st.dataframe(st_safe(df), use_container_width=True)


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
