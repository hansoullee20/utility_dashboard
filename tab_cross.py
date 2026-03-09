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
from lang import t

_BLDG_COLOR_MAP = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}

_Z_THRESH = 2.0   # |z| above this → anomaly flag


def _bar(df: pd.DataFrame, x: str, y: str, title: str, y_label: str,
         color_col: str | None = "building", key: str | None = None):
    fig = px.bar(
        df, x=x, y=y,
        color=color_col if color_col and color_col in df.columns else None,
        title=title,
        labels={y: y_label, x: "Brand"},
        color_discrete_map=_BLDG_COLOR_MAP,
    )
    fig.update_layout(height=420, xaxis_tickangle=-45, showlegend=True,
                      margin=dict(t=50, b=80))
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    _chart_key = key or f"cross_bar_{title[:30].replace(' ', '_')}"
    _ev = st.plotly_chart(fig, use_container_width=True, key=_chart_key, on_select="rerun")
    _sel = _ev.selection.points if _ev and hasattr(_ev, "selection") else []
    if _sel:
        _pt = _sel[0]
        _brand = _pt.get("x") or _pt.get("customdata") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = df[df[x] == _brand] if _brand and x in df.columns else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)


def _flag_anomalies(df: pd.DataFrame, z_col: str, label_col: str) -> pd.DataFrame:
    """Return rows where |z| >= threshold, sorted by z descending."""
    if z_col not in df.columns:
        return pd.DataFrame()
    mask = df[z_col].abs() >= _Z_THRESH
    out = df.loc[mask].copy().sort_values(z_col, ascending=False)
    return out


# ── Section 1: Unit cost ──────────────────────────────────────────────────────

def _render_unit_costs(unit_df: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader(t("cross_unit_title"))
    st.caption(t("cross_unit_cap"))

    tabs = []
    if "water_unit_cost" in unit_df.columns:
        tabs.append(("💧 Water", "water_unit_cost", "water_unit_z", "₩/m³"))
    if "elect_unit_cost" in unit_df.columns:
        tabs.append(("⚡ Electricity", "elect_unit_cost", "elect_unit_z", "₩/kWh"))
    if "total_cost_per_m2" in unit_df.columns:
        tabs.append(("📊 Total/m²", "total_cost_per_m2", "total_cost_per_m2_z", "만원/m²"))

    if not tabs:
        st.info("No unit cost data available.")
        return

    tab_labels = [t[0] for t in tabs]
    st_tabs = st.tabs(tab_labels)

    for st_tab, (label, val_col, z_col, unit) in zip(st_tabs, tabs):
        with st_tab:
            plot_df = unit_df.dropna(subset=[val_col]).sort_values(val_col, ascending=False)
            if plot_df.empty:
                st.info(f"No data for {label}.")
                continue

            _bar(plot_df, x="brand", y=val_col,
                 title=f"{label} Unit Cost ({unit})", y_label=unit,
                 color_col="building" if split_by_building else None,
                 key=f"cross_unit_{val_col}")

            # Annotate anomalies
            anomalies = _flag_anomalies(plot_df, z_col, "brand")
            if not anomalies.empty:
                st.warning(f"**{len(anomalies)} {t('cross_anomaly')}** |z| ≥ {_Z_THRESH:.0f} — {label}")
                disp_cols = [c for c in ["brand", "building", val_col, z_col] if c in anomalies.columns]
                st.dataframe(anomalies[disp_cols], hide_index=True, use_container_width=True)

            # Full table
            with st.expander(t("cross_full_table"), expanded=False):
                disp_cols = [c for c in [
                    "brand", "building", "size_m2",
                    "water_usage_m3", val_col, z_col,
                ] if c in plot_df.columns]
                view = add_display_index(plot_df[disp_cols])
                st.dataframe(view, hide_index=True, use_container_width=True)
                download_df_as_excel(view,
                                     filename=f"unit_cost_{val_col}.xlsx",
                                     sheet_name="unit_cost")


# ── Section 2: Electricity breakdown ─────────────────────────────────────────

def _render_elec_breakdown(elec_br: pd.DataFrame, split_by_building: bool = True) -> None:
    st.subheader(t("cross_elec_title"))
    st.caption(t("cross_elec_cap"))

    pct_cols = {c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in elec_br.columns}
    if not pct_cols:
        st.info("Electricity breakdown data not available.")
        return

    # Stacked bar: melt pct columns
    melt_cols = [c for c in ["ehp_pct", "hvac_pct", "base_pct"] if c in elec_br.columns]
    melt_df = elec_br[["brand"] + melt_cols].melt(id_vars="brand", var_name="category", value_name="pct")
    label_map = {"ehp_pct": "EHP", "hvac_pct": "HVAC (EHP+FCU+AHU)", "base_pct": "Base Load"}
    melt_df["category"] = melt_df["category"].map(label_map)

    fig = px.bar(
        melt_df, x="brand", y="pct", color="category",
        title="Electricity Category Share (%)",
        labels={"pct": "% of total kWh", "brand": "Brand"},
        barmode="group",
    )
    fig.update_layout(height=420, xaxis_tickangle=-45, margin=dict(t=50, b=80))
    _ev_elec_br = st.plotly_chart(fig, use_container_width=True, key="cross_elec_breakdown_stacked", on_select="rerun")
    _sel_elec_br = _ev_elec_br.selection.points if _ev_elec_br and hasattr(_ev_elec_br, "selection") else []
    if _sel_elec_br:
        _pt = _sel_elec_br[0]
        _brand = _pt.get("x") or _pt.get("customdata") or ""
        if isinstance(_brand, (list, tuple)):
            _brand = _brand[0]
        _fdf = elec_br[elec_br["brand"] == _brand] if _brand else pd.DataFrame()
        if not _fdf.empty:
            st.caption(f"선택됨: **{_brand}**")
            st.dataframe(_fdf.reset_index(drop=True), hide_index=True, use_container_width=True)

    # HVAC intensity bar
    if "hvac_intensity" in elec_br.columns:
        intensity_df = elec_br.dropna(subset=["hvac_intensity"]).sort_values("hvac_intensity", ascending=False)
        _bar(intensity_df, x="brand", y="hvac_intensity",
             title="HVAC Intensity (kWh/m²)",
             y_label="kWh/m²",
             color_col="building" if split_by_building else None,
             key="cross_hvac_intensity")

    # Electricity unit cost from elec sheet
    if "elect_unit_cost" in elec_br.columns:
        uc_df = elec_br.dropna(subset=["elect_unit_cost"]).sort_values("elect_unit_cost", ascending=False)
        _bar(uc_df, x="brand", y="elect_unit_cost",
             title="Electricity Unit Cost from Detail Sheet (₩/kWh)",
             y_label="₩/kWh",
             color_col="building" if split_by_building else None,
             key="cross_elect_unit_cost_detail")

    with st.expander(t("cross_elec_full"), expanded=False):
        show_cols = [c for c in [
            "brand", "building", "kwh_total", "kwh_ehp", "ehp_pct",
            "kwh_hvac", "hvac_pct", "kwh_base", "base_pct",
            "hvac_intensity", "elect_unit_cost",
        ] if c in elec_br.columns]
        view = add_display_index(elec_br[show_cols].sort_values("kwh_total", ascending=False))
        st.dataframe(view, hide_index=True, use_container_width=True)
        download_df_as_excel(view, filename="elec_breakdown.xlsx", sheet_name="elec_breakdown")


# ── Public render ─────────────────────────────────────────────────────────────

def render_cross_tab(
    cur_df: pd.DataFrame,
    file_name: str,
    file_data: bytes,
    sheet_names: list[str],
    split_by_building: bool = True,
) -> None:
    """Render the cross-sheet feature engineering tab.

    Loads billing + electricity detail on user request (lazy), then shows
    unit cost analysis, electricity breakdown, and anomaly flags.
    """
    has_billing = BILLING_SHEET_NAME in sheet_names
    has_elec    = ELECTRICITY_SHEET_NAME in sheet_names

    if not has_billing and not has_elec:
        st.info(t("cross_no_sheets"))
        return

    _key = f"cross_loaded_{file_name}"

    if not st.session_state.get(_key):
        available = []
        if has_billing:
            available.append(f"`{BILLING_SHEET_NAME}`")
        if has_elec:
            available.append(f"`{ELECTRICITY_SHEET_NAME}`")
        st.info(f"{t('cross_avail')}: {', '.join(available)}. {t('cross_load_btn')} 버튼을 클릭하세요." if st.session_state.get('lang','ko') == 'ko' else f"{t('cross_avail')}: {', '.join(available)}. Click below to load.")
        if st.button(t("cross_load_btn"), key="btn_load_cross"):
            st.session_state[_key] = True
            st.rerun()
        return

    with st.spinner(t("cross_loading")):
        billing_df = elec_df = None
        errors = []

        if has_billing:
            try:
                billing_df = read_billing_sheet(file_name, file_data, BILLING_SHEET_NAME)
            except Exception as e:
                errors.append(f"Billing sheet: {e}")

        if has_elec:
            try:
                elec_df = read_electricity_sheet(file_name, file_data, ELECTRICITY_SHEET_NAME)
            except Exception as e:
                errors.append(f"Electricity sheet: {e}")

    for err in errors:
        st.warning(f"Could not load {err}")

    # ── Unit cost section ─────────────────────────────────────────────────────
    if billing_df is not None and not billing_df.empty:
        try:
            unit_df = build_unit_costs(cur_df, billing_df)
            if not unit_df.empty:
                _render_unit_costs(unit_df, split_by_building=split_by_building)
                st.divider()
        except Exception as e:
            st.warning(f"{t('cross_unit_fail')}: {e}")

    # ── Electricity breakdown section ─────────────────────────────────────────
    if elec_df is not None and not elec_df.empty:
        try:
            elec_br = build_elec_breakdown(elec_df, meter_df=cur_df)
            if not elec_br.empty:
                _render_elec_breakdown(elec_br, split_by_building=split_by_building)
        except Exception as e:
            st.warning(f"{t('cross_elec_fail')}: {e}")

    if billing_df is None and elec_df is None:
        st.error(t("cross_no_data"))
