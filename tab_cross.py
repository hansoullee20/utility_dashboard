"""tab_cross.py — Cross-sheet feature engineering tab.

Joins 검침내역 + 수도광열비 부과 내역 + 전체 전기 사용내역 to display:
  1. Unit cost analysis (₩/m³, ₩/kWh) with Z-score anomaly flags
  2. Electricity breakdown (EHP%, HVAC%, base load%)
  3. Total cost per m² ranking
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from data import (
    to_numeric_series,
    read_billing_sheet, read_electricity_sheet,
    BILLING_SHEET_NAME, ELECTRICITY_SHEET_NAME,
)
from cross_features import build_unit_costs, build_elec_breakdown
from features import add_display_index, download_df_as_excel
from biz_report import render_pdf_buttons, generate_cross_pdf
from utils_plot import bar_chart
from lang import t

_Z_THRESH = 2.0


def _flag_anomalies(df: pd.DataFrame, z_col: str) -> pd.DataFrame:
    if z_col not in df.columns:
        return pd.DataFrame()
    return df.loc[df[z_col].abs() >= _Z_THRESH].copy().sort_values(z_col, ascending=False)


# ── Section 1: Unit cost ──────────────────────────────────────────────────────

def _render_unit_costs(unit_df: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader(t("cross_unit_title"))
    st.caption(t("cross_unit_cap"))

    tabs_spec = []
    for col, label, unit in [
        ("water_unit_cost", "💧 Water",    "₩/m³"),
        ("elect_unit_cost", "⚡ Electricity", "₩/kWh"),
        ("total_cost_per_m2", "📊 Total/m²", "만원/m²"),
    ]:
        if col in unit_df.columns:
            tabs_spec.append((col, label, unit))

    if not tabs_spec:
        st.info("No unit cost data available.")
        return

    for st_tab, (val_col, label, unit) in zip(
        st.tabs([s[1] for s in tabs_spec]), tabs_spec
    ):
        z_col = val_col + "_z" if val_col != "total_cost_per_m2" else "total_cost_per_m2_z"
        with st_tab:
            plot_df = unit_df.dropna(subset=[val_col]).sort_values(val_col, ascending=False)
            if plot_df.empty:
                st.info(f"No data for {label}.")
                continue

            bar_chart(
                plot_df, x="brand", y=val_col,
                title=f"{label} Unit Cost ({unit})", y_label=unit,
                color_col="building" if split_by_building else None,
                key=f"cross_unit_{val_col}",
            )

            anomalies = _flag_anomalies(plot_df, z_col)
            if not anomalies.empty:
                st.warning(f"**{len(anomalies)} {t('cross_anomaly')}** |z| ≥ {_Z_THRESH:.0f} — {label}")
                disp_cols = [c for c in ["brand", "building", val_col, z_col] if c in anomalies.columns]
                st.dataframe(anomalies[disp_cols], hide_index=True, use_container_width=True)

            with st.expander(t("cross_full_table"), expanded=False):
                disp_cols = [c for c in ["brand", "building", "size_m2", "water_usage_m3", val_col, z_col]
                             if c in plot_df.columns]
                view = add_display_index(plot_df[disp_cols])
                st.dataframe(view, hide_index=True, use_container_width=True)
                download_df_as_excel(view, filename=f"unit_cost_{val_col}.xlsx", sheet_name="unit_cost")


# ── Section 2: Electricity breakdown ─────────────────────────────────────────

def _render_elec_breakdown(elec_br: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader(t("cross_elec_title"))
    st.caption(t("cross_elec_cap"))

    pct_cols = [c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in elec_br.columns]
    if not pct_cols:
        st.info("Electricity breakdown data not available.")
        return

    melt_df = elec_br[["brand"] + pct_cols].melt(id_vars="brand", var_name="category", value_name="pct")
    melt_df["category"] = melt_df["category"].map(
        {"ehp_pct": "EHP", "hvac_pct": "HVAC (EHP+FCU+AHU)", "base_pct": "Base Load"}
    )
    fig = px.bar(melt_df, x="brand", y="pct", color="category", barmode="group",
                 title="Electricity Category Share (%)",
                 labels={"pct": "% of total kWh", "brand": "Brand"})
    fig.update_layout(height=420, xaxis_tickangle=-45, margin=dict(t=50, b=80))
    _ev = st.plotly_chart(fig, use_container_width=True, key="cross_elec_breakdown_stacked", on_select="rerun")
    _pts = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _pts:
        _brand = (_pts[0].get("x") or "")
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = elec_br[elec_br["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    if "hvac_intensity" in elec_br.columns:
        bar_chart(
            elec_br.dropna(subset=["hvac_intensity"]).sort_values("hvac_intensity", ascending=False),
            x="brand", y="hvac_intensity",
            title="HVAC Intensity (kWh/m²)", y_label="kWh/m²",
            color_col="building" if split_by_building else None,
            key="cross_hvac_intensity",
        )

    if "elect_unit_cost" in elec_br.columns:
        bar_chart(
            elec_br.dropna(subset=["elect_unit_cost"]).sort_values("elect_unit_cost", ascending=False),
            x="brand", y="elect_unit_cost",
            title="Electricity Unit Cost from Detail Sheet (₩/kWh)", y_label="₩/kWh",
            color_col="building" if split_by_building else None,
            key="cross_elect_unit_cost_detail",
        )

    with st.expander(t("cross_elec_full"), expanded=False):
        show_cols = [c for c in [
            "brand", "building", "kwh_total", "kwh_ehp", "ehp_pct",
            "kwh_hvac", "hvac_pct", "kwh_base", "base_pct",
            "hvac_intensity", "elect_unit_cost",
        ] if c in elec_br.columns]
        view = add_display_index(elec_br[show_cols].sort_values("kwh_total", ascending=False))
        st.dataframe(view, hide_index=True, use_container_width=True)
        download_df_as_excel(view, filename="elec_breakdown.xlsx", sheet_name="elec_breakdown")


# ── YoY comparison helper ─────────────────────────────────────────────────────

def _render_yoy_cross(cur_df, unit_df, yoy_df,
                      yoy_file, yoy_file_data, yoy_sheet_names,
                      split_by_building=True,
                      billing_period=None, yoy_billing_period=None):
    """Show YoY changes in unit costs."""
    _period_str = f"{yoy_billing_period} → {billing_period}" if billing_period and yoy_billing_period else "전년 대비"
    st.subheader(f"📅 전년 대비 단위 비용 변화  ({_period_str})")

    # Build YoY unit costs
    yoy_billing_df = None
    if BILLING_SHEET_NAME in yoy_sheet_names:
        try:
            yoy_billing_df = read_billing_sheet(yoy_file, yoy_file_data, BILLING_SHEET_NAME)
        except Exception:
            pass

    if yoy_billing_df is None or yoy_billing_df.empty:
        st.info("전년 부과 내역 시트가 없습니다.")
        return

    try:
        yoy_unit_df = build_unit_costs(yoy_df, yoy_billing_df)
        if yoy_unit_df.empty:
            st.info("전년 단위 비용 데이터를 생성할 수 없습니다.")
            return
    except Exception:
        st.info("전년 단위 비용 계산 중 오류가 발생했습니다.")
        return

    if unit_df is None:
        st.info("당월 단위 비용 데이터가 없습니다.")
        return

    # Merge current and YoY unit costs by brand
    _cost_cols = [c for c in ["water_unit_cost", "elect_unit_cost", "total_cost_per_m2"]
                  if c in unit_df.columns and c in yoy_unit_df.columns]
    if not _cost_cols:
        st.info("비교 가능한 단위 비용 항목이 없습니다.")
        return

    _cur = unit_df[["brand"] + _cost_cols].copy()
    _yoy = yoy_unit_df[["brand"] + _cost_cols].copy()
    _merged = _cur.merge(_yoy, on="brand", suffixes=("", "_yoy"), how="inner")

    if _merged.empty:
        st.info("전년 대비 매칭되는 브랜드가 없습니다.")
        return

    _labels = {"water_unit_cost": "💧 수도 (₩/m³)", "elect_unit_cost": "⚡ 전기 (₩/kWh)",
               "total_cost_per_m2": "📊 총비용 (만원/m²)"}

    # KPI row
    _kc = st.columns(len(_cost_cols))
    for i, col in enumerate(_cost_cols):
        _c_med = _merged[col].median()
        _y_med = _merged[f"{col}_yoy"].median()
        _d = _c_med - _y_med
        _pct = _d / _y_med * 100 if _y_med else 0
        _kc[i].metric(
            _labels.get(col, col),
            f"{_c_med:,.2f}",
            delta=f"{_d:+,.2f} ({_pct:+.1f}%)",
            delta_color="inverse",
            help="중앙값 기준 전년 대비 변화",
        )

    # Per-brand change table
    for col in _cost_cols:
        chg_col = f"{col}_변화"
        _merged[chg_col] = _merged[col] - _merged[f"{col}_yoy"]
    _disp_cols = ["brand"] + [c for col in _cost_cols for c in [col, f"{col}_yoy", f"{col}_변화"]]
    _disp = _merged[[c for c in _disp_cols if c in _merged.columns]].copy()
    _rename = {}
    for col in _cost_cols:
        lbl = _labels.get(col, col)
        _rename[col] = f"올해 {lbl}"
        _rename[f"{col}_yoy"] = f"전년 {lbl}"
        _rename[f"{col}_변화"] = f"변화 {lbl}"
    _disp = _disp.rename(columns=_rename)
    st.dataframe(_disp.round(2), hide_index=True, use_container_width=True)


# ── Public render ─────────────────────────────────────────────────────────────

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
    has_billing = BILLING_SHEET_NAME in sheet_names
    has_elec    = ELECTRICITY_SHEET_NAME in sheet_names

    if not has_billing and not has_elec:
        st.info(t("cross_no_sheets"))
        return

    with st.spinner(t("cross_loading")):
        billing_df = elec_df = None
        if has_billing:
            try:
                billing_df = read_billing_sheet(file_name, file_data, BILLING_SHEET_NAME)
            except Exception as e:
                st.warning(f"Billing sheet: {e}")
        if has_elec:
            try:
                elec_df = read_electricity_sheet(file_name, file_data, ELECTRICITY_SHEET_NAME)
            except Exception as e:
                st.warning(f"Electricity sheet: {e}")

    unit_df = elec_br = None
    if billing_df is not None and not billing_df.empty:
        try:
            unit_df = build_unit_costs(cur_df, billing_df)
            if unit_df.empty:
                unit_df = None
        except Exception as e:
            st.warning(f"{t('cross_unit_fail')}: {e}")

    if elec_df is not None and not elec_df.empty:
        try:
            elec_br = build_elec_breakdown(elec_df, meter_df=cur_df)
            if elec_br.empty:
                elec_br = None
        except Exception as e:
            st.warning(f"{t('cross_elec_fail')}: {e}")

    if unit_df is None and elec_br is None:
        st.error(t("cross_no_data"))
        return

    # ── Sections ──────────────────────────────────────────────────────────────
    if unit_df is not None:
        _render_unit_costs(unit_df, split_by_building=split_by_building)
        st.divider()

    if elec_br is not None:
        _render_elec_breakdown(elec_br, split_by_building=split_by_building)
        st.divider()

    # ── YoY comparison ──────────────────────────────────────────────────────
    if yoy_df is not None and yoy_file and yoy_file_data:
        _render_yoy_cross(
            cur_df, unit_df, yoy_df,
            yoy_file, yoy_file_data, yoy_sheet_names or [],
            split_by_building=split_by_building,
            billing_period=billing_period,
            yoy_billing_period=yoy_billing_period,
        )
        st.divider()

    # ── Reference — PDF + raw data ───────────────────────────────────────────
    _pdf_key = f"cross_pdf_{file_name}"
    render_pdf_buttons(
        _pdf_key,
        lambda: generate_cross_pdf(unit_df, elec_br),
        "📥 비용분석 리포트",
        "비용분석_리포트.pdf",
    )

    with st.expander("📊 원시 데이터", expanded=False):
        if unit_df is not None:
            st.markdown("**단위 비용**")
            st.dataframe(unit_df, hide_index=True, use_container_width=True)
        if elec_br is not None:
            st.markdown("**전기 분해**")
            st.dataframe(elec_br, hide_index=True, use_container_width=True)
