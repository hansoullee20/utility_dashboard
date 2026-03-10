"""tab_efficiency.py — Energy efficiency ranking tab."""
import pandas as pd
import plotly.express as px
import streamlit as st

from data import to_numeric_series, build_ehp_analysis
from features import add_display_index, download_df_as_excel
from lang import t

_UTIL_LABELS = {
    "water":  "💧 수도",
    "hwater": "🌡️ 온수",
    "elect":  "⚡ 전기",
    "heat":   "🔥 난방",
}
_UNIT_LABELS = {
    "water":  "m³/m²",
    "hwater": "m³/m²",
    "elect":  "kWh/m²",
    "heat":   "m³(MWh)/m²",
}
_BLDG_COLOR_MAP = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "D": "#9467bd"}


def _bar(df: pd.DataFrame, x: str, y: str, title: str, y_label: str,
         split_by_building: bool = True, key: str | None = None) -> None:
    fig = px.bar(
        df, x=x, y=y,
        color="building" if split_by_building and "building" in df.columns else None,
        title=title,
        labels={y: y_label, x: "Brand"},
        color_discrete_map=_BLDG_COLOR_MAP,
    )
    fig.update_layout(height=420, xaxis_tickangle=-45, showlegend=True, margin=dict(t=50, b=80))
    fig.update_traces(marker_line_width=0.5, marker_line_color="white")
    _chart_key = key or f"eff_bar_{title[:30].replace(' ', '_')}"
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


def _render_single_utility(cur_df: pd.DataFrame, avail: dict[str, str],
                            split_by_building: bool = True) -> None:
    st.subheader(t("eff_single_title"))

    sel = st.selectbox(
        t("eff_utility_sel"), list(avail.keys()),
        format_func=lambda p: _UTIL_LABELS.get(p, p),
        key="eff_util_select",
    )
    per_m2_col = avail[sel]
    per_py_col = f"{sel}_usage_per_py"
    curr_col   = f"{sel}_current"

    detail_cols = [c for c in ["brand", "building", "floor", "size_m2", "size_py",
                                curr_col, per_m2_col, per_py_col]
                   if c in cur_df.columns]

    eff_df = (
        cur_df[detail_cols].copy()
        .dropna(subset=[per_m2_col])
        .sort_values(per_m2_col, ascending=False)
        .reset_index(drop=True)
    )
    for c in [per_m2_col, per_py_col]:
        if c in eff_df.columns:
            eff_df[c] = eff_df[c].round(4)

    unit = _UNIT_LABELS.get(sel, "unit/m²")
    _bar(eff_df, x="brand", y=per_m2_col,
         title=f"{_UTIL_LABELS.get(sel, sel)} — Usage per m² ({unit})",
         y_label=unit, split_by_building=split_by_building,
         key=f"eff_single_{sel}_per_m2")

    eff_view = add_display_index(eff_df)
    st.dataframe(eff_view, hide_index=True, use_container_width=True)
    download_df_as_excel(eff_view, filename=f"efficiency_{sel}_per_m2.xlsx", sheet_name="efficiency")


def _render_ehp(cur_df: pd.DataFrame, ehp_annual: pd.DataFrame,
                split_by_building: bool = True) -> None:
    """Show EHP (HVAC) electricity consumption per m² by brand."""
    st.subheader(t("eff_ehp_title"))

    # Find the most recent year column with data
    year_cols = sorted([c for c in ehp_annual.columns if str(c).isdigit()], reverse=True)
    latest = next((y for y in year_cols if ehp_annual[y].notna().any()), None)
    if latest is None:
        st.info(t("eff_no_ehp"))
        return

    # Aggregate meters by brand for the latest year
    ehp_by_brand = (
        ehp_annual.groupby("brand")[latest]
        .sum()
        .reset_index()
        .rename(columns={latest: "ehp_kwh"})
    )

    # Join with size data from cur_df
    size_df = (
        cur_df[["brand", "building", "size_m2"]]
        .groupby(["brand", "building"], as_index=False)["size_m2"]
        .first()
    )
    ehp_merged = ehp_by_brand.merge(size_df, on="brand", how="inner")
    ehp_merged = ehp_merged[to_numeric_series(ehp_merged["size_m2"]) > 0].copy()
    ehp_merged["ehp_per_m2"] = (
        to_numeric_series(ehp_merged["ehp_kwh"]) /
        to_numeric_series(ehp_merged["size_m2"])
    ).round(4)
    ehp_merged = ehp_merged.dropna(subset=["ehp_per_m2"]).sort_values("ehp_per_m2", ascending=False).reset_index(drop=True)

    if ehp_merged.empty:
        st.info(t("eff_no_ehp_match"))
        return

    st.caption(f"Based on {latest} annual EHP usage (kWh) across all meters per brand.")
    _bar(ehp_merged, x="brand", y="ehp_per_m2",
         title=f"EHP Usage per m² — {latest} (kWh/m²)",
         y_label="kWh/m²", split_by_building=split_by_building,
         key=f"eff_ehp_per_m2_{latest}")

    ehp_view = add_display_index(ehp_merged[["brand", "building", "size_m2", "ehp_kwh", "ehp_per_m2"]])
    st.dataframe(ehp_view, hide_index=True, use_container_width=True)
    download_df_as_excel(ehp_view, filename=f"efficiency_ehp_{latest}.xlsx", sheet_name="ehp_efficiency")

    return ehp_merged  # returned so combined score can use it


def _render_combined(cur_df: pd.DataFrame, avail: dict[str, str], ehp_merged: pd.DataFrame | None,
                     split_by_building: bool = True) -> None:
    st.subheader(t("eff_combined_title"))
    st.caption(t("eff_combined_cap"))

    id_cols = [c for c in ["brand", "building", "floor", "size_m2"] if c in cur_df.columns]
    combined = cur_df[id_cols + list(avail.values())].copy().dropna(subset=list(avail.values()), how="all")

    norm_cols = []
    for p, col in avail.items():
        s = to_numeric_series(combined[col])
        rng = s.max() - s.min()
        norm_col = f"{p}_norm"
        combined[norm_col] = ((s - s.min()) / rng).round(4) if rng > 0 else 0.0
        norm_cols.append(norm_col)
        combined[col] = combined[col].round(4)

    # Optionally include EHP
    if ehp_merged is not None and not ehp_merged.empty:
        ehp_norm_src = ehp_merged[["brand", "ehp_per_m2"]].copy()
        combined = combined.merge(ehp_norm_src, on="brand", how="left")
        s = to_numeric_series(combined["ehp_per_m2"])
        rng = s.max() - s.min()
        combined["ehp_norm"] = ((s - s.min()) / rng).round(4) if rng > 0 else 0.0
        norm_cols.append("ehp_norm")

    if len(norm_cols) < 2:
        st.info(t("eff_need_two"))
        return

    combined["efficiency_score"] = combined[norm_cols].mean(axis=1, skipna=True).round(4)
    combined = (
        combined.drop(columns=norm_cols + (["ehp_per_m2"] if "ehp_per_m2" in combined.columns else []))
        .sort_values("efficiency_score", ascending=False)
        .reset_index(drop=True)
    )
    combined.insert(0, "Rank", range(1, len(combined) + 1))

    _bar(combined, x="brand", y="efficiency_score",
         title="Combined Efficiency Score (higher = more consumption per m²)",
         y_label="Score [0–1]", split_by_building=split_by_building,
         key="eff_combined_score")

    combined_view = add_display_index(combined.drop(columns=["Rank"]))
    st.dataframe(combined_view, hide_index=True, use_container_width=True)
    download_df_as_excel(combined_view, filename="efficiency_combined.xlsx", sheet_name="efficiency_combined")


def render_efficiency_tab(
    cur_df: pd.DataFrame,
    present: list[str],
    file_name: str | None = None,
    file_data: bytes | None = None,
    ehp_sheet: str | None = None,
    split_by_building: bool = True,
) -> None:
    """Rank brands by per-area current usage to evaluate energy efficiency."""
    avail = {p: f"{p}_usage_per_m2" for p in present if f"{p}_usage_per_m2" in cur_df.columns}

    if not avail and ehp_sheet is None:
        st.info(t("eff_no_size"))
        return

    if avail:
        _render_single_utility(cur_df, avail, split_by_building=split_by_building)
        st.divider()

    # ── EHP section: lazy-load on user request ────────────────────────────────
    ehp_merged = None
    if ehp_sheet and file_name and file_data:
        _ehp_key = f"ehp_loaded_{file_name}"
        if not st.session_state.get(_ehp_key):
            if st.button(t("load_ehp_btn"), key="btn_load_ehp"):
                st.session_state[_ehp_key] = True
                st.rerun()
        else:
            try:
                with st.spinner(t("ehp_spinner")):
                    _, ehp_annual = build_ehp_analysis(file_name, file_data, ehp_sheet)
                if ehp_annual is not None and not ehp_annual.empty:
                    ehp_merged = _render_ehp(cur_df, ehp_annual, split_by_building=split_by_building)
                    st.divider()
            except Exception as e:
                st.warning(f"{t('ehp_load_fail')}: {e}")

    if avail:
        _render_combined(cur_df, avail, ehp_merged, split_by_building=split_by_building)

    # ── PDF download ──────────────────────────────────────────────────────────
    if avail and file_name:
        st.divider()
        _pdf_key = f"eff_pdf_{file_name}"
        _col_gen, _col_dl = st.columns([1, 2])
        with _col_gen:
            if st.button("📄 PDF 리포트 생성", key=f"gen_eff_pdf_{file_name}"):
                with st.spinner("PDF 생성 중…"):
                    from biz_report import generate_efficiency_pdf
                    st.session_state[_pdf_key] = generate_efficiency_pdf(cur_df, present)
        if _pdf_key in st.session_state:
            with _col_dl:
                st.download_button(
                    "⬇️ 효율분析 리포트 다운로드",
                    st.session_state[_pdf_key],
                    file_name="효율분析_리포트.pdf",
                    mime="application/pdf",
                    key=f"dl_eff_pdf_{file_name}",
                )
