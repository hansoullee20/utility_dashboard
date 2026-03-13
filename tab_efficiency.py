"""tab_efficiency.py — Energy efficiency ranking tab."""
import pandas as pd
import streamlit as st

from data import to_numeric_series, build_ehp_analysis
from features import add_display_index, download_df_as_excel
from biz_report import render_pdf_buttons, generate_efficiency_pdf
from utils_plot import bar_chart
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
    bar_chart(
        eff_df, x="brand", y=per_m2_col,
        title=f"{_UTIL_LABELS.get(sel, sel)} — Usage per m² ({unit})",
        y_label=unit,
        color_col="building" if split_by_building else None,
        key=f"eff_single_{sel}_per_m2",
    )

    eff_view = add_display_index(eff_df)
    st.dataframe(eff_view, hide_index=True, use_container_width=True)
    download_df_as_excel(eff_view, filename=f"efficiency_{sel}_per_m2.xlsx", sheet_name="efficiency")


def _render_ehp(cur_df: pd.DataFrame, ehp_annual: pd.DataFrame,
                split_by_building: bool = True) -> pd.DataFrame | None:
    st.subheader(t("eff_ehp_title"))

    year_cols = sorted([c for c in ehp_annual.columns if str(c).isdigit()], reverse=True)
    latest = next((y for y in year_cols if ehp_annual[y].notna().any()), None)
    if latest is None:
        st.info(t("eff_no_ehp"))
        return None

    ehp_by_brand = (
        ehp_annual.groupby("brand")[latest].sum().reset_index().rename(columns={latest: "ehp_kwh"})
    )
    size_df = (
        cur_df[["brand", "building", "size_m2"]]
        .groupby(["brand", "building"], as_index=False)["size_m2"].first()
    )
    ehp_merged = ehp_by_brand.merge(size_df, on="brand", how="inner")
    ehp_merged = ehp_merged[to_numeric_series(ehp_merged["size_m2"]) > 0].copy()
    ehp_merged["ehp_per_m2"] = (
        to_numeric_series(ehp_merged["ehp_kwh"]) / to_numeric_series(ehp_merged["size_m2"])
    ).round(4)
    ehp_merged = (ehp_merged.dropna(subset=["ehp_per_m2"])
                             .sort_values("ehp_per_m2", ascending=False)
                             .reset_index(drop=True))

    if ehp_merged.empty:
        st.info(t("eff_no_ehp_match"))
        return None

    st.caption(f"Based on {latest} annual EHP usage (kWh) across all meters per brand.")
    bar_chart(
        ehp_merged, x="brand", y="ehp_per_m2",
        title=f"EHP Usage per m² — {latest} (kWh/m²)", y_label="kWh/m²",
        color_col="building" if split_by_building else None,
        key=f"eff_ehp_per_m2_{latest}",
    )

    ehp_view = add_display_index(ehp_merged[["brand", "building", "size_m2", "ehp_kwh", "ehp_per_m2"]])
    st.dataframe(ehp_view, hide_index=True, use_container_width=True)
    download_df_as_excel(ehp_view, filename=f"efficiency_ehp_{latest}.xlsx", sheet_name="ehp_efficiency")

    # ── EHP multi-year trend chart ─────────────────────────────────────────
    avail_years = sorted([y for y in year_cols if ehp_annual[y].notna().any()])
    if len(avail_years) >= 2:
        st.divider()
        st.subheader("📈 EHP 연간 추세")
        st.caption(f"{avail_years[0]}–{avail_years[-1]} 연간 EHP 사용량 추이 (kWh)")

        # Build brand-level annual totals across all years — top 10 by total
        brand_year = ehp_annual.groupby("brand")[avail_years].sum()
        top_brands = brand_year.loc[brand_year.sum(axis=1).nlargest(10).index]

        import plotly.graph_objects as _go
        fig_trend = _go.Figure()
        for brand in top_brands.index:
            vals = top_brands.loc[brand]
            fig_trend.add_trace(_go.Scatter(
                x=[str(y) for y in avail_years],
                y=vals.values,
                mode="lines+markers",
                name=str(brand)[:20],
                hovertemplate=f"<b>{brand}</b><br>%{{x}}: %{{y:,.0f}} kWh<extra></extra>",
            ))
        fig_trend.update_layout(
            title=f"EHP 연간 사용량 추이 — 상위 10개 브랜드 (kWh)",
            height=420,
            xaxis_title="연도",
            yaxis_title="사용량 (kWh)",
            margin=dict(t=55, b=60, l=60, r=20),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5,
                        font=dict(size=9)),
            hovermode="x unified",
        )
        _ev_trend = st.plotly_chart(fig_trend, use_container_width=True, key="eff_ehp_trend", on_select="rerun")

        with st.expander("📋 연간 추세 데이터", expanded=False):
            trend_view = top_brands.reset_index()
            trend_view.columns = ["brand"] + [str(y) for y in avail_years]
            st.dataframe(trend_view, hide_index=True, use_container_width=True)

    return ehp_merged


def _render_combined(cur_df: pd.DataFrame, avail: dict[str, str],
                     ehp_merged: pd.DataFrame | None, split_by_building: bool = True) -> None:
    st.subheader(t("eff_combined_title"))
    st.caption(t("eff_combined_cap"))

    id_cols  = [c for c in ["brand", "building", "floor", "size_m2"] if c in cur_df.columns]
    combined = cur_df[id_cols + list(avail.values())].copy().dropna(subset=list(avail.values()), how="all")

    norm_cols = []
    for p, col in avail.items():
        s   = to_numeric_series(combined[col])
        rng = s.max() - s.min()
        norm_col = f"{p}_norm"
        combined[norm_col] = ((s - s.min()) / rng).round(4) if rng > 0 else 0.0
        norm_cols.append(norm_col)
        combined[col] = combined[col].round(4)

    if ehp_merged is not None and not ehp_merged.empty:
        combined = combined.merge(ehp_merged[["brand", "ehp_per_m2"]], on="brand", how="left")
        s   = to_numeric_series(combined["ehp_per_m2"])
        rng = s.max() - s.min()
        combined["ehp_norm"] = ((s - s.min()) / rng).round(4) if rng > 0 else 0.0
        norm_cols.append("ehp_norm")

    if len(norm_cols) < 2:
        st.info(t("eff_need_two"))
        return

    combined["efficiency_score"] = combined[norm_cols].mean(axis=1, skipna=True).round(4)
    drop_cols = norm_cols + (["ehp_per_m2"] if "ehp_per_m2" in combined.columns else [])
    combined  = combined.drop(columns=drop_cols).sort_values("efficiency_score", ascending=False).reset_index(drop=True)
    combined.insert(0, "Rank", range(1, len(combined) + 1))

    bar_chart(
        combined, x="brand", y="efficiency_score",
        title="Combined Efficiency Score (higher = more consumption per m²)",
        y_label="Score [0–1]",
        color_col="building" if split_by_building else None,
        key="eff_combined_score",
    )

    combined_view = add_display_index(combined.drop(columns=["Rank"]))
    st.dataframe(combined_view, hide_index=True, use_container_width=True)
    download_df_as_excel(combined_view, filename="efficiency_combined.xlsx", sheet_name="efficiency_combined")


# ── Public render ─────────────────────────────────────────────────────────────

def _render_yoy_efficiency(cur_df, yoy_df, avail,
                          split_by_building=True,
                          billing_period=None, yoy_billing_period=None):
    """Show YoY changes in per-m² usage efficiency."""
    _period_str = f"{yoy_billing_period} → {billing_period}" if billing_period and yoy_billing_period else "전년 대비"
    st.subheader(f"📅 전년 대비 효율 변화  ({_period_str})")

    # Find matching per_m2 columns in both current and YoY
    _yoy_avail = {p: col for p, col in avail.items() if col in yoy_df.columns}
    if not _yoy_avail:
        st.info("전년 효율 데이터가 없습니다.")
        return

    _cur = cur_df[["brand"] + [col for col in _yoy_avail.values()]].copy()
    _yoy = yoy_df[["brand"] + [col for col in _yoy_avail.values()]].copy()
    _merged = _cur.merge(_yoy, on="brand", suffixes=("", "_yoy"), how="inner")

    if _merged.empty:
        st.info("전년 대비 매칭되는 브랜드가 없습니다.")
        return

    # KPI row — median per-m² usage change
    _kc = st.columns(len(_yoy_avail))
    for i, (p, col) in enumerate(_yoy_avail.items()):
        _c_med = _merged[col].median()
        _y_med = _merged[f"{col}_yoy"].median()
        _d = _c_med - _y_med
        _pct = _d / _y_med * 100 if _y_med else 0
        _unit = _UNIT_LABELS.get(p, "unit/m²")
        _kc[i].metric(
            f"{_UTIL_LABELS.get(p, p)} ({_unit})",
            f"{_c_med:,.4f}",
            delta=f"{_d:+,.4f} ({_pct:+.1f}%)",
            delta_color="inverse",
            help="중앙값 기준 전년 대비 변화",
        )

    # Per-brand change table
    for p, col in _yoy_avail.items():
        chg_col = f"{col}_변화"
        pct_col = f"{col}_변화율"
        _merged[chg_col] = (_merged[col] - _merged[f"{col}_yoy"]).round(4)
        _merged[pct_col] = (
            (_merged[chg_col] / _merged[f"{col}_yoy"].replace(0, float("nan"))) * 100
        ).round(1)

    _disp_cols = ["brand"]
    _rename = {}
    for p, col in _yoy_avail.items():
        lbl = _UTIL_LABELS.get(p, p)
        _disp_cols += [col, f"{col}_yoy", f"{col}_변화", f"{col}_변화율"]
        _rename[col] = f"올해 {lbl}"
        _rename[f"{col}_yoy"] = f"전년 {lbl}"
        _rename[f"{col}_변화"] = f"변화 {lbl}"
        _rename[f"{col}_변화율"] = f"변화율(%) {lbl}"
    _disp = _merged[[c for c in _disp_cols if c in _merged.columns]].copy()
    _disp = _disp.rename(columns=_rename)
    st.dataframe(_disp, hide_index=True, use_container_width=True)


def render_efficiency_tab(
    cur_df: pd.DataFrame,
    present: list[str],
    file_name: str | None = None,
    file_data: bytes | None = None,
    ehp_sheet: str | None = None,
    split_by_building: bool = True,
    yoy_df: pd.DataFrame | None = None,
    billing_period: str | None = None,
    yoy_billing_period: str | None = None,
    sheet_names: list[str] | None = None,
) -> None:
    """Rank brands by per-area current usage to evaluate energy efficiency."""
    avail = {p: f"{p}_usage_per_m2" for p in present if f"{p}_usage_per_m2" in cur_df.columns}

    if not avail and ehp_sheet is None:
        st.info(t("eff_no_size"))
        return

    # ── Business Insight Summary ────────────────────────────────────────────
    _eff_insights: list[str] = []
    _PLABEL = {"water": "수도", "hwater": "온수", "elect": "전기", "heat": "난방"}
    from data import to_numeric_series as _tns_eff
    for p, pm2_col in avail.items():
        s = _tns_eff(cur_df[pm2_col]).dropna()
        if s.empty:
            continue
        _worst = cur_df.loc[s.idxmax()]
        _best = cur_df.loc[s.idxmin()]
        _ratio = s.max() / s.min() if s.min() > 0 else float("inf")
        if _ratio >= 3:
            _eff_insights.append(
                f"**{_PLABEL.get(p, p)}** m²당: "
                f"최고 {_worst['brand']}({s.max():.2f}) vs "
                f"최저 {_best['brand']}({s.min():.2f}) — "
                f"**{_ratio:.1f}배** 차이 → "
                f"{_worst['brand']}의 사용 패턴을 점검하세요. 누수·장비 이상 가능성이 있습니다"
            )
        else:
            _eff_insights.append(
                f"**{_PLABEL.get(p, p)}** m²당: "
                f"최고 {_worst['brand']}({s.max():.2f}) vs "
                f"최저 {_best['brand']}({s.min():.2f}) — "
                f"**{_ratio:.1f}배** 차이"
            )
    if _eff_insights:
        # Overall: how many brands are above median?
        for p, pm2_col in avail.items():
            s = _tns_eff(cur_df[pm2_col]).dropna()
            if s.empty:
                continue
            _med = s.median()
            _n_above = int((s > _med * 1.5).sum())
            if _n_above:
                _eff_insights.append(
                    f"{_PLABEL.get(p, p)} 중위값 대비 1.5배 초과: **{_n_above}개** 브랜드 → "
                    f"에너지 낭비 가능성이 높아 절감 조치를 우선 검토하세요"
                )
            break  # just show for first utility to keep it concise

        with st.container(border=True):
            st.markdown(
                '<p style="margin:0 0 6px;font-size:0.9rem;font-weight:700;color:#4C72B0">'
                '비즈니스 인사이트</p>',
                unsafe_allow_html=True,
            )
            st.markdown("  \n".join(f"- {i}" for i in _eff_insights))

    # ── Sections ──────────────────────────────────────────────────────────────
    if avail:
        _render_single_utility(cur_df, avail, split_by_building=split_by_building)
        st.divider()

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

    # ── YoY efficiency comparison ─────────────────────────────────────────────
    if yoy_df is not None and avail:
        st.divider()
        _render_yoy_efficiency(cur_df, yoy_df, avail,
                               split_by_building=split_by_building,
                               billing_period=billing_period,
                               yoy_billing_period=yoy_billing_period)

    # ── Reference — PDF + raw data ───────────────────────────────────────────
    st.divider()
    if avail and file_name:
        _pdf_key = f"eff_pdf_{file_name}"
        render_pdf_buttons(
            _pdf_key,
            lambda: generate_efficiency_pdf(cur_df, present),
            "📥 효율분석 리포트",
            "효율분석_리포트.pdf",
        )

    with st.expander("📊 원시 데이터", expanded=False):
        eff_cols = [c for c in ["brand", "building", "floor", "size_m2", "size_py"]
                    + list(avail.values())
                    + [f"{p}_current" for p in present]
                    if c in cur_df.columns]
        st.dataframe(cur_df[eff_cols], hide_index=True, use_container_width=True)
