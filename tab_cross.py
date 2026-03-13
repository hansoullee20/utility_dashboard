"""tab_cross.py — Cross-sheet feature engineering tab (비용 분석).

Joins 검침내역 + 수도광열비 부과 내역 + 전체 전기 사용내역 to display:
  1. KPI overview — median unit costs + anomaly count
  2. Unit cost analysis (₩/m³, ₩/kWh, 만원/m²) with Z-score anomaly flags
  3. Electricity breakdown (EHP%, HVAC%, base load%)
  4. YoY cost comparison (when available)
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import (
    to_numeric_series,
    read_billing_sheet, read_electricity_sheet,
    BILLING_SHEET_NAME, ELECTRICITY_SHEET_NAME,
)
from cross_features import build_elec_breakdown
from features import add_display_index, download_df_as_excel
from biz_report import render_pdf_buttons, generate_cross_pdf
from utils_plot import bar_chart
from utils import fmt_won
from lang import t

_Z_THRESH = 2.0

_BLDG_COLOR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}

_COST_META = {
    "water_unit_cost":   {"label": "💧 수도 단가",  "unit": "₩/m³",   "z": "water_unit_z"},
    "elect_unit_cost":   {"label": "⚡ 전기 단가",  "unit": "₩/kWh",  "z": "elect_unit_z"},
    "total_cost_per_py": {"label": "📊 평당 비용",  "unit": "만원/평", "z": "total_cost_per_py_z"},
    "total_cost_per_m2": {"label": "📊 총비용/m²", "unit": "만원/m²", "z": "total_cost_per_m2_z"},
}

# Per-utility billing breakdown metadata
_UTIL_COST_META = [
    {"col": "water_total",   "label": "💧 수도",      "unit": "만원"},
    {"col": "elect_total",   "label": "⚡ 전기",      "unit": "만원"},
    {"col": "heat_total",    "label": "🔥 난방",      "unit": "만원"},
    {"col": "hotwater_cost", "label": "🌡️ 온수",     "unit": "만원"},
    {"col": "hvac_cost",     "label": "🔧 HVAC",     "unit": "만원"},
    {"col": "total",         "label": "📊 총비용",    "unit": "만원"},
    {"col": "cost_per_m2",   "label": "📐 면적당 총비용", "unit": "만원/m²", "is_ratio": True},
]


def _flag_anomalies(df: pd.DataFrame, z_col: str) -> pd.DataFrame:
    if z_col not in df.columns:
        return pd.DataFrame()
    return df.loc[df[z_col].abs() >= _Z_THRESH].copy().sort_values(z_col, ascending=False)


# ── Section 0: KPI overview ─────────────────────────────────────────────────

def _render_cost_kpis(billing_df: pd.DataFrame | None) -> None:
    if billing_df is None:
        st.info("비용 데이터가 없습니다.")
        return

    bill_agg = _prepare_billing_by_brand(billing_df)
    items = []
    for m in _UTIL_COST_META:
        col = m["col"]
        if col not in bill_agg.columns:
            continue
        s = bill_agg[col].dropna()
        if s.empty:
            continue
        med = s.median()
        z_col = f"{col}_z"
        n_anom = int((bill_agg[z_col].abs() >= _Z_THRESH).sum()) if z_col in bill_agg.columns else 0
        is_ratio = m.get("is_ratio", False)
        items.append((m["label"], med, n_anom, is_ratio))

    if not items:
        st.info("비용 데이터가 없습니다.")
        return

    kpi = st.container(border=True)
    with kpi:
        cols = st.columns(min(len(items), 6))
        for col_w, (label, val, n_anom, is_ratio) in zip(cols, items):
            if is_ratio:
                val_str = f"{val:,.2f} 만원/m²"
            else:
                val_str = fmt_won(val * 10000)
            delta_str = f"이상치 {n_anom}개" if n_anom > 0 else None
            col_w.metric(label, val_str,
                         delta=delta_str,
                         delta_color="inverse" if n_anom and n_anom > 0 else "off",
                         help="중앙값 기준")
        st.caption(f"📂 {len(bill_agg)}개 브랜드 · 분석 데이터: **청구**")


# ── Section 1: Unit cost ─────────────────────────────────────────────────────

def _prepare_billing_by_brand(billing_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate billing data per brand+building, add derived columns and Z-scores."""
    import numpy as np
    # Add hotwater/hvac combined columns
    df = billing_df.copy()
    hw_cols = [c for c in ["hotwater_excl", "hotwater_comm"] if c in df.columns]
    if hw_cols:
        df["hotwater_cost"] = df[hw_cols].sum(axis=1)
    hv_cols = [c for c in ["hvac_excl", "hvac_comm"] if c in df.columns]
    if hv_cols:
        df["hvac_cost"] = df[hv_cols].sum(axis=1)

    # Aggregate
    group = ["brand", "building"] if "building" in df.columns else ["brand"]
    num_cols = [m["col"] for m in _UTIL_COST_META if m["col"] in df.columns and m["col"] != "cost_per_m2"]
    if "size_m2" in df.columns:
        num_cols.append("size_m2")
    num_cols = list(dict.fromkeys(num_cols))
    agg = df.groupby(group, as_index=False)[num_cols].sum(min_count=1)

    # Compute cost_per_m2
    if "total" in agg.columns and "size_m2" in agg.columns:
        size = to_numeric_series(agg["size_m2"]).replace(0, np.nan)
        agg["cost_per_m2"] = (agg["total"] / size).round(4)

    # Z-scores for each cost column
    for m in _UTIL_COST_META:
        col = m["col"]
        if col in agg.columns:
            s = agg[col].dropna()
            if len(s) >= 3:
                mean, std = s.mean(), s.std()
                agg[f"{col}_z"] = (((agg[col] - mean) / std).round(2) if std > 0
                                   else 0.0)
    return agg


def _render_unit_costs(billing_df: pd.DataFrame, split_by_building: bool = True,
                       selected_util: dict | None = None) -> None:
    st.subheader("💰 단가 분석")
    st.caption("유틸리티별 비용 분석. 또래 평균 대비 등급: 매우높음 / 높음 / 보통 / 낮음 / 매우낮음")

    bill_agg = _prepare_billing_by_brand(billing_df)

    if selected_util is None:
        st.info("유틸리티별 비용 데이터 없음")
        return

    meta = selected_util
    val_col = meta["col"]
    label = meta["label"]
    unit = meta["unit"]
    z_col = f"{val_col}_z"

    plot_df = bill_agg.dropna(subset=[val_col]).sort_values(val_col, ascending=False)
    if plot_df.empty:
        st.info(f"{label} 데이터 없음")
        return

    s = plot_df[val_col]
    _mean, _med, _std = s.mean(), s.median(), s.std()
    _min, _max = s.min(), s.max()
    _cv = (_std / _mean * 100) if _mean else 0
    _is_ratio = meta.get("is_ratio", False)
    if _is_ratio:
        _vfmt = lambda v: f"{v:,.2f} {unit}"
    else:
        _vfmt = lambda v: fmt_won(v * 10000)

    anomalies = _flag_anomalies(plot_df, z_col) if z_col in plot_df.columns else pd.DataFrame()

    # ── Interpretation box ─────────────────────────────────────────────
    if _cv > 50:
        st.warning(f"⚠️ 비용 편차가 매우 큼 (CV {_cv:.0f}%) — 과금 오류 또는 계약 조건 차이 가능성")
    elif _cv > 30:
        st.info(f"📌 비용 편차가 다소 큼 (CV {_cv:.0f}%) — 사용 패턴 차이 확인 필요")
    else:
        st.success(f"비용이 비교적 균일 (CV {_cv:.0f}%)")

    # ── Bar chart + stats + anomaly in one row ─────────────────────────
    _c_bar, _c_stats, _c_anom = st.columns([3, 1.5, 1.5])
    with _c_bar:
        bar_chart(
            plot_df, x="brand", y=val_col,
            title=f"{label} 비용 ({unit})", y_label=unit,
            color_col="building" if split_by_building else None,
            key=f"cross_util_{val_col}",
            height=360,
        )
    with _c_stats:
        st.markdown("**📊 기초 통계**")
        stats_md = (
            f"| 통계 | 값 |\n|---|---|\n"
            f"| 평균 | {_vfmt(_mean)} |\n"
            f"| 중앙값 | {_vfmt(_med)} |\n"
            f"| 표준편차 | {_vfmt(_std)} |\n"
            f"| 범위 | {_vfmt(_min)} ~ {_vfmt(_max)} |\n"
            f"| 변동계수(CV) | {_cv:.1f}% |"
        )
        st.markdown(stats_md)
    with _c_anom:
        if not anomalies.empty:
            from utils import z_col_to_badge as _z2b
            st.markdown(f"**⚠️ 이상치 {len(anomalies)}개**")
            _anom_cols = [c for c in ["brand", "building", val_col, z_col] if c in anomalies.columns]
            _anom_disp = anomalies[_anom_cols].copy()
            if z_col in _anom_disp.columns:
                _anom_disp[z_col] = _z2b(_anom_disp[z_col])
            _ren = {"brand": "브랜드", "building": "건물",
                    val_col: f"비용 ({unit})", z_col: "등급"}
            st.dataframe(
                _anom_disp.rename(columns=_ren).reset_index(drop=True),
                hide_index=True, use_container_width=True,
            )
        else:
            st.success("이상치 없음")

    # ── Histogram + stats + anomaly in one row ────────────────────────
    from viz import plot_hist_with_tails as _plot_hist
    _bin_key = f"cross_hist_bins_{val_col}"
    _iqr_key = f"cross_hist_iqr_{val_col}"
    if _bin_key not in st.session_state:
        st.session_state[_bin_key] = 50
    if _iqr_key not in st.session_state:
        st.session_state[_iqr_key] = 1.5
    _h_bins = st.session_state[_bin_key]
    _iqr_k = st.session_state[_iqr_key]
    _q1, _q3 = s.quantile(0.25), s.quantile(0.75)
    _iqr = _q3 - _q1
    _lo = _q1 - _iqr_k * _iqr
    _hi = _q3 + _iqr_k * _iqr
    _c_hist, _c_hstats, _c_hanom = st.columns([3, 1.5, 1.5])
    with _c_hist:
        _plot_hist(
            s, bins=_h_bins, lo=float(_lo), hi=float(_hi),
            title=f"{label} 비용 분포 ({unit})",
            key=f"cross_hist_{val_col}",
            source_df=plot_df, val_col=val_col, val_scale=1.0,
            display_cols=["brand", "building", val_col, z_col] if z_col in plot_df.columns else ["brand", "building", val_col],
            show_bins_slider=False,
            show_stats_row=False,
            show_outlier_list=False,
        )
    with _c_hstats:
        st.markdown("**📈 분포 통계**")
        _p20 = float(s.quantile(0.20))
        _p80 = float(s.quantile(0.80))
        _n = int(s.notna().sum())
        hist_stats_md = (
            f"| 통계 | 값 |\n|---|---|\n"
            f"| n | {_n} |\n"
            f"| 최소 | {_vfmt(_min)} |\n"
            f"| p20 | {_vfmt(_p20)} |\n"
            f"| 중앙값 | {_vfmt(_med)} |\n"
            f"| 평균 | {_vfmt(_mean)} |\n"
            f"| 표준편차 | {_vfmt(_std)} |\n"
            f"| p80 | {_vfmt(_p80)} |\n"
            f"| 최대 | {_vfmt(_max)} |"
        )
        st.markdown(hist_stats_md)
    with _c_hanom:
        if not anomalies.empty:
            st.markdown(f"**⚠️ 이상치 {len(anomalies)}개** — |Z| ≥ {_Z_THRESH:.0f}")
            _anom_cols = [c for c in ["brand", "building", val_col, z_col] if c in anomalies.columns]
            _ren = {"brand": "브랜드", "building": "건물",
                    val_col: f"비용 ({unit})", z_col: "Z-점수"}
            st.dataframe(
                anomalies[_anom_cols].rename(columns=_ren).reset_index(drop=True),
                hide_index=True, use_container_width=True,
            )
        else:
            st.success("이상치 없음")

    # ── Histogram controls + IQR equation ──────────────────────────────
    _ctrl1, _ctrl2 = st.columns(2)
    with _ctrl1:
        st.slider("Bins", 5, 200, key=_bin_key, step=5)
    with _ctrl2:
        st.slider("IQR 배수 (k)", 0.5, 3.0, key=_iqr_key, step=0.25,
                  help="이상치 기준: Q1 − k×IQR  /  Q3 + k×IQR")
    st.markdown(
        f"$$Q_1 = {_q1:,.2f},\\quad Q_3 = {_q3:,.2f},\\quad IQR = {_iqr:,.2f}$$\n\n"
        f"$$\\text{{Lower}} = Q_1 - {_iqr_k}\\times IQR = {_lo:,.2f}"
        f",\\quad \\text{{Upper}} = Q_3 + {_iqr_k}\\times IQR = {_hi:,.2f}$$"
    )

    # ── IQR outlier list ───────────────────────────────────────────────
    _top_iqr = plot_df[s > _hi]
    _bot_iqr = plot_df[s < _lo]
    _n_top_iqr = len(_top_iqr)
    _n_bot_iqr = len(_bot_iqr)
    _n_out_iqr = _n_top_iqr + _n_bot_iqr
    _out_disp = [c for c in ["brand", "building", val_col, z_col] if c in plot_df.columns]
    _out_ren = {"brand": "브랜드", "building": "건물",
                val_col: f"비용 ({unit})", z_col: "Z-점수"}
    with st.expander(f"🔶 이상치 목록 — {_n_out_iqr}건  (상위 {_n_top_iqr} · 하위 {_n_bot_iqr})",
                     expanded=_n_out_iqr > 0):
        if _n_out_iqr == 0:
            st.caption("이상치 없음")
        else:
            if _n_top_iqr > 0:
                st.markdown(f"**🔺 상위 이상치** — {_hi:.4g} 초과 ({_n_top_iqr}건)")
                st.dataframe(
                    _top_iqr[_out_disp].rename(columns=_out_ren)
                    .sort_values(f"비용 ({unit})", ascending=False).reset_index(drop=True),
                    hide_index=True, use_container_width=True,
                )
            if _n_bot_iqr > 0:
                if _n_top_iqr > 0:
                    st.divider()
                st.markdown(f"**🔻 하위 이상치** — {_lo:.4g} 미만 ({_n_bot_iqr}건)")
                st.dataframe(
                    _bot_iqr[_out_disp].rename(columns=_out_ren)
                    .sort_values(f"비용 ({unit})", ascending=True).reset_index(drop=True),
                    hide_index=True, use_container_width=True,
                )

    with st.expander("전체 테이블", expanded=False):
        _disp_cols = [c for c in ["brand", "building", "size_m2", val_col, z_col]
                     if c in plot_df.columns]
        view = add_display_index(plot_df[_disp_cols])
        st.dataframe(view, hide_index=True, use_container_width=True)
        download_df_as_excel(view, filename=f"util_cost_{val_col}.xlsx", sheet_name="비용")


# ── Section: Building comparison (uses shared selector) ──────────────────────

def _render_building_comparison(billing_df: pd.DataFrame, selected_util: dict) -> None:
    bill_agg = _prepare_billing_by_brand(billing_df)
    val_col = selected_util["col"]
    label = selected_util["label"]
    unit = selected_util["unit"]
    _is_ratio = selected_util.get("is_ratio", False)

    if val_col not in bill_agg.columns:
        return
    plot_df = bill_agg.dropna(subset=[val_col])
    if "building" not in plot_df.columns or plot_df["building"].nunique() <= 1:
        return

    st.subheader(f"🏢 건물별 비교 — {label}")
    bldg_agg = plot_df.groupby("building", as_index=False).agg(
        합계=(val_col, "sum"),
        평균=(val_col, "mean"),
        중앙값=(val_col, "median"),
        브랜드수=(val_col, "count"),
    ).sort_values("합계", ascending=False)

    _bc1, _bc2 = st.columns([3, 2])
    with _bc1:
        bar_chart(
            bldg_agg, x="building", y="합계",
            title=f"{label} 건물별 합계 ({unit})", y_label=unit,
            color_col="building",
            key=f"cross_bldg_{val_col}",
            height=320,
            show_logy=False,
        )
    with _bc2:
        disp = bldg_agg.copy()
        if _is_ratio:
            for c in ["합계", "평균", "중앙값"]:
                disp[c] = disp[c].apply(lambda v: f"{v:,.2f}")
        else:
            for c in ["합계", "평균", "중앙값"]:
                disp[c] = disp[c].apply(lambda v: fmt_won(v * 10000))
        disp = disp.rename(columns={"building": "건물"})
        st.dataframe(disp, hide_index=True, use_container_width=True)


# ── Section: Electricity breakdown ───────────────────────────────────────────

def _render_elec_breakdown(elec_br: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader("⚡ 전기 사용 카테고리 분류")
    st.caption("브랜드별 전기 소비 카테고리 비율. **HVAC** = EHP + FCU + AHU. **기본** = 일반 + 펌프 + 주방 환풍기.")

    pct_cols = [c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in elec_br.columns]
    if not pct_cols:
        st.info("전기 분류 데이터 없음")
        return

    # Stats summary for HVAC ratio
    if "hvac_pct" in elec_br.columns:
        _hvac = elec_br["hvac_pct"].dropna()
        if len(_hvac) > 1:
            _hm, _hmed, _hstd = _hvac.mean(), _hvac.median(), _hvac.std()
            _high_hvac = int((_hvac > 60).sum())
            stats_box = st.container(border=True)
            with stats_box:
                _sc = st.columns(4)
                _sc[0].metric("HVAC 평균 비중", f"{_hm:.1f}%")
                _sc[1].metric("HVAC 중앙값", f"{_hmed:.1f}%")
                _sc[2].metric("HVAC 표준편차", f"{_hstd:.1f}%")
                _sc[3].metric("HVAC > 60%", f"{_high_hvac}개",
                              delta=f"{_high_hvac}개 고비중" if _high_hvac else None,
                              delta_color="inverse")
            if _hmed > 50:
                st.warning("⚠️ HVAC가 전체 전기 사용의 절반 이상 — 냉난방 효율 점검 권장")
            elif _hmed > 35:
                st.info("📌 HVAC 비중이 높은 편 — 계절별 패턴 확인 필요")

    # Stacked bar chart
    melt_df = elec_br[["brand"] + pct_cols].melt(
        id_vars="brand", var_name="category", value_name="pct")
    melt_df["category"] = melt_df["category"].map(
        {"ehp_pct": "EHP", "hvac_pct": "HVAC", "base_pct": "기본 부하"}
    )
    fig = px.bar(melt_df, x="brand", y="pct", color="category", barmode="stack",
                 title="전기 카테고리 비율 (%)",
                 labels={"pct": "비율 (%)", "brand": "브랜드"},
                 color_discrete_map={"EHP": "#DD8A00", "HVAC": "#C44E52", "기본 부하": "#4C72B0"})
    fig.update_layout(height=400, xaxis_tickangle=-45, margin=dict(t=50, b=80))
    _ev = st.plotly_chart(fig, use_container_width=True, key="cross_elec_stacked", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = (_pts[0].get("x") or "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = elec_br[elec_br["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # HVAC intensity + elect unit cost side by side
    _has_hvac = "hvac_intensity" in elec_br.columns
    _has_elec = "elect_unit_cost" in elec_br.columns
    if _has_hvac or _has_elec:
        _cols = st.columns(2 if _has_hvac and _has_elec else 1)
        _ci = 0
        if _has_hvac:
            with _cols[_ci]:
                bar_chart(
                    elec_br.dropna(subset=["hvac_intensity"]).sort_values("hvac_intensity", ascending=False),
                    x="brand", y="hvac_intensity",
                    title="HVAC 강도 (kWh/m²)", y_label="kWh/m²",
                    color_col="building" if split_by_building else None,
                    key="cross_hvac_intensity", height=360,
                )
            _ci += 1
        if _has_elec:
            with _cols[_ci]:
                bar_chart(
                    elec_br.dropna(subset=["elect_unit_cost"]).sort_values("elect_unit_cost", ascending=False),
                    x="brand", y="elect_unit_cost",
                    title="전기 단가 (₩/kWh)", y_label="₩/kWh",
                    color_col="building" if split_by_building else None,
                    key="cross_elect_unit_cost_detail", height=360,
                )

    with st.expander("📋 전체 전기 분류 테이블", expanded=False):
        show_cols = [c for c in [
            "brand", "building", "kwh_total", "kwh_ehp", "ehp_pct",
            "kwh_hvac", "hvac_pct", "kwh_base", "base_pct",
            "hvac_intensity", "elect_unit_cost",
        ] if c in elec_br.columns]
        view = add_display_index(elec_br[show_cols].sort_values("kwh_total", ascending=False))
        st.dataframe(view, hide_index=True, use_container_width=True)
        download_df_as_excel(view, filename="elec_breakdown.xlsx", sheet_name="전기분류")


# ── Section: Water breakdown ─────────────────────────────────────────────────

def _render_water_breakdown(water_br: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader("💧 수도 요금 카테고리 분류")
    st.caption("브랜드별 수도 요금 구성 비율. **수도요금** = 상수도, **하수도** = 하수도 요금, **부담금** = 물이용부담금, **관경비** = 관경 공용비.")

    pct_cols = [c for c in ["water_pct", "sewage_pct", "levy_pct", "pipe_pct"] if c in water_br.columns]
    if not pct_cols:
        st.info("수도 분류 데이터 없음")
        return

    # Stats summary for sewage ratio
    if "sewage_pct" in water_br.columns:
        _sw = water_br["sewage_pct"].dropna()
        if len(_sw) > 1:
            _sm, _smed, _sstd = _sw.mean(), _sw.median(), _sw.std()
            _high_sw = int((_sw > 50).sum())
            stats_box = st.container(border=True)
            with stats_box:
                _sc = st.columns(4)
                _sc[0].metric("하수도 평균 비중", f"{_sm:.1f}%")
                _sc[1].metric("하수도 중앙값", f"{_smed:.1f}%")
                _sc[2].metric("하수도 표준편차", f"{_sstd:.1f}%")
                _sc[3].metric("하수도 > 50%", f"{_high_sw}개",
                              delta=f"{_high_sw}개 고비중" if _high_sw else None,
                              delta_color="inverse")

    # Stacked bar chart
    _label_map = {
        "water_pct": "수도요금", "sewage_pct": "하수도",
        "levy_pct": "부담금", "pipe_pct": "관경비",
    }
    _color_map = {
        "수도요금": "#4C72B0", "하수도": "#C44E52",
        "부담금": "#DD8A00", "관경비": "#2ca02c",
    }
    melt_df = water_br[["brand"] + pct_cols].melt(
        id_vars="brand", var_name="category", value_name="pct")
    melt_df["category"] = melt_df["category"].map(_label_map)
    fig = px.bar(melt_df, x="brand", y="pct", color="category", barmode="stack",
                 title="수도 요금 카테고리 비율 (%)",
                 labels={"pct": "비율 (%)", "brand": "브랜드"},
                 color_discrete_map=_color_map)
    fig.update_layout(height=400, xaxis_tickangle=-45, margin=dict(t=50, b=80))
    _ev = st.plotly_chart(fig, use_container_width=True, key="cross_water_stacked", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = (_pts[0].get("x") or "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = water_br[water_br["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # Water intensity + avg unit price side by side
    _has_wi = "water_intensity" in water_br.columns
    _has_up = "avg_unit_price" in water_br.columns
    if _has_wi or _has_up:
        _cols = st.columns(2 if _has_wi and _has_up else 1)
        _ci = 0
        if _has_wi:
            with _cols[_ci]:
                bar_chart(
                    water_br.dropna(subset=["water_intensity"]).sort_values("water_intensity", ascending=False),
                    x="brand", y="water_intensity",
                    title="수도 사용 강도 (m³/m²)", y_label="m³/m²",
                    color_col="building" if split_by_building else None,
                    key="cross_water_intensity", height=360,
                )
            _ci += 1
        if _has_up:
            with _cols[_ci]:
                bar_chart(
                    water_br.dropna(subset=["avg_unit_price"]).sort_values("avg_unit_price", ascending=False),
                    x="brand", y="avg_unit_price",
                    title="수도 평균 단가 (₩/m³)", y_label="₩/m³",
                    color_col="building" if split_by_building else None,
                    key="cross_water_avg_unit", height=360,
                )

    with st.expander("📋 전체 수도 분류 테이블", expanded=False):
        show_cols = [c for c in [
            "brand", "building", "usage_m3", "water_fee", "sewage_fee",
            "levy_fee", "pipe_fee_comm", "total",
            "water_pct", "sewage_pct", "levy_pct", "pipe_pct",
            "water_intensity", "avg_unit_price",
        ] if c in water_br.columns]
        view = add_display_index(water_br[show_cols].sort_values("total", ascending=False))
        st.dataframe(view, hide_index=True, use_container_width=True)
        download_df_as_excel(view, filename="water_breakdown.xlsx", sheet_name="수도분류")


# ── YoY comparison helper ────────────────────────────────────────────────────

def _render_yoy_cross(billing_df, yoy_file, yoy_file_data, yoy_sheet_names,
                      split_by_building=True,
                      billing_period=None, yoy_billing_period=None,
                      selected_util=None):
    """Show YoY changes in per-utility billing costs."""
    _period_str = f"{yoy_billing_period} → {billing_period}" if billing_period and yoy_billing_period else "전년 대비"
    st.subheader(f"📅 전년 대비 유틸리티 비용 변화  ({_period_str})")

    # Load YoY billing
    _yoy_bill_key = next(
        (s for s in (yoy_sheet_names or []) if s.strip() == BILLING_SHEET_NAME), None,
    )
    if not _yoy_bill_key or not yoy_file_data:
        st.info("전년 부과 내역 시트가 없습니다.")
        return

    try:
        yoy_billing_df = read_billing_sheet(yoy_file, yoy_file_data, _yoy_bill_key)
    except Exception as _e:
        st.warning(f"전년 청구 시트 로드 실패: {_e}")
        return

    if yoy_billing_df is None or yoy_billing_df.empty:
        return

    cur_agg = _prepare_billing_by_brand(billing_df)
    yoy_agg = _prepare_billing_by_brand(yoy_billing_df)

    # Find common utility cost columns
    _cost_cols = [m["col"] for m in _UTIL_COST_META
                  if m["col"] in cur_agg.columns and m["col"] in yoy_agg.columns
                  and cur_agg[m["col"]].notna().any() and yoy_agg[m["col"]].notna().any()]
    if not _cost_cols:
        st.info("비교 가능한 비용 항목이 없습니다.")
        return

    _cur = cur_agg[["brand"] + _cost_cols].copy()
    _yoy = yoy_agg[["brand"] + _cost_cols].copy()
    _merged = _cur.merge(_yoy, on="brand", suffixes=("", "_yoy"), how="inner")

    if _merged.empty:
        st.info("전년 대비 매칭되는 브랜드가 없습니다.")
        return

    # KPI row — per-utility median change
    _meta_map = {m["col"]: m for m in _UTIL_COST_META}
    kpi = st.container(border=True)
    with kpi:
        _kc = st.columns(len(_cost_cols))
        for i, col in enumerate(_cost_cols):
            m = _meta_map[col]
            _c_med = _merged[col].median()
            _y_med = _merged[f"{col}_yoy"].median()
            _d = _c_med - _y_med
            _pct = _d / _y_med * 100 if _y_med else 0
            _is_r = m.get("is_ratio", False)
            if _is_r:
                _v_str = f"{_c_med:,.2f} {m['unit']}"
                _d_str = f"{_d:+,.2f} ({_pct:+.1f}%)"
            else:
                _v_str = fmt_won(_c_med * 10000)
                _d_str = f"{fmt_won(_d * 10000, signed=True)} ({_pct:+.1f}%)"
            _kc[i].metric(
                m["label"], _v_str,
                delta=_d_str,
                delta_color="inverse",
                help="중앙값 기준 전년 대비 변화",
            )

    # Use shared utility selection
    if selected_util is None or selected_util["col"] not in _cost_cols:
        col = _cost_cols[0]
        m = _meta_map[col]
    else:
        col = selected_util["col"]
        m = _meta_map[col]

    chg_col = f"{col}_변화"
    pct_col = f"{col}_변화율"
    _merged[chg_col] = _merged[col] - _merged[f"{col}_yoy"]
    _merged[pct_col] = (
        _merged[chg_col] / _merged[f"{col}_yoy"].replace(0, float("nan")) * 100
    ).round(1)

    # Bar chart — change per brand
    _is_r = m.get("is_ratio", False)
    _sorted = _merged.sort_values(chg_col, ascending=False).reset_index(drop=True)
    _colors = _sorted[chg_col].apply(lambda v: "#C44E52" if v > 0 else "#2ca02c").tolist()
    if _is_r:
        _y_vals = _sorted[chg_col]
        _text = _sorted[chg_col].apply(lambda v: f"{v:+,.2f}")
        _ytitle = f"변화 ({m['unit']})"
    else:
        _y_vals = _sorted[chg_col] * 10000
        _text = _sorted[chg_col].apply(lambda v: fmt_won(v * 10000, signed=True))
        _ytitle = "변화 (원)"
    fig = go.Figure(go.Bar(
        x=[str(b)[:18] for b in _sorted["brand"]],
        y=_y_vals,
        marker_color=_colors,
        text=_text,
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="<b>%{x}</b><br>변화: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{m['label']} 전년 대비 변화",
        height=430, xaxis_tickangle=-45,
        yaxis_title=_ytitle,
        margin=dict(t=55, b=80, l=60, r=20),
        showlegend=False,
    )
    fig.add_hline(y=0, line_color="#888888", line_width=1)
    _ev = st.plotly_chart(fig, use_container_width=True, key=f"cross_yoy_{col}_bar", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = _pts[0].get("x", "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = _sorted[_sorted["brand"].astype(str).str[:18] == str(_brand)[:18]]
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # Per-brand change table
    _disp = _sorted[["brand", f"{col}_yoy", col, chg_col, pct_col]].copy()
    _disp = _disp.rename(columns={
        "brand": "브랜드",
        f"{col}_yoy": f"전년 {m['label']}",
        col: f"올해 {m['label']}",
        chg_col: "변화",
        pct_col: "변화율(%)",
    })
    st.dataframe(_disp.round(2), hide_index=True, use_container_width=True)


# ── Public render ────────────────────────────────────────────────────────────

def render_cross_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    sheet_names: list[str],
    split_by_building: bool = True,
    yoy_df: pd.DataFrame | None = None,
    yoy_file: str | None = None,
    yoy_file_data: bytes | None = None,
    yoy_sheet_names: list[str] | None = None,
    billing_period: str | None = None,
    yoy_billing_period: str | None = None,
) -> None:
    _find = lambda target: next((s for s in sheet_names if s.strip() == target), None)
    _bill_key = _find(BILLING_SHEET_NAME)

    if not _bill_key:
        st.info(t("cross_no_sheets"))
        return

    with st.spinner(t("cross_loading")):
        billing_df = None
        if _bill_key:
            try:
                billing_df = read_billing_sheet(file_name, file_data, _bill_key)
            except Exception as e:
                st.warning(f"청구 시트: {e}")

    if billing_df is None:
        st.error(t("cross_no_data"))
        return

    # ── 0. KPI overview ────────────────────────────────────────────────────
    _render_cost_kpis(billing_df)

    # ── Shared utility selector ──────────────────────────────────────────
    _selected_util = None
    if billing_df is not None:
        _bill_agg = _prepare_billing_by_brand(billing_df)
        _avail_utils = [m for m in _UTIL_COST_META if m["col"] in _bill_agg.columns
                        and _bill_agg[m["col"]].notna().any()]
        if _avail_utils:
            import streamlit_antd_components as _sac_top
            _util_labels = [m["label"] for m in _avail_utils]
            _util_sel = _sac_top.segmented(
                [_sac_top.SegmentedItem(label=lbl) for lbl in _util_labels],
                key="cross_util_sel", use_container_width=True,
            )
            _sel_i = _util_labels.index(_util_sel) if _util_sel in _util_labels else 0
            _selected_util = _avail_utils[_sel_i]

    # ── 1. Per-utility cost breakdown (uses shared selector) ─────────────
    if billing_df is not None and _selected_util is not None:
        _render_unit_costs(billing_df, split_by_building=split_by_building,
                           selected_util=_selected_util)

    # ── 2. YoY comparison (uses shared selector) ─────────────────────────
    if billing_df is not None and yoy_file and yoy_file_data and _selected_util is not None:
        _render_yoy_cross(
            billing_df,
            yoy_file, yoy_file_data, yoy_sheet_names or [],
            split_by_building=split_by_building,
            billing_period=billing_period,
            yoy_billing_period=yoy_billing_period,
            selected_util=_selected_util,
        )

    # ── 3. Building comparison (uses shared selector) ────────────────────
    if billing_df is not None and _selected_util is not None:
        _render_building_comparison(billing_df, _selected_util)

    st.divider()
    # ── 4. Reference ──────────────────────────────────────────────────────
    _pdf_key = f"cross_pdf_{file_name}"
    render_pdf_buttons(
        _pdf_key,
        lambda: generate_cross_pdf(
            _prepare_billing_by_brand(billing_df) if billing_df is not None else None,
            None,
        ),
        "📥 비용분석 리포트",
        "비용분석_리포트.pdf",
    )
