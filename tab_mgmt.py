"""tab_mgmt.py — 경영 보고 sub-tab (moved from summary.py to 점검대상)."""
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import BLD_COLOR as _BLD_COLOR, fmt_won as _fmt_won, iqr_upper as _iqr_upper
from utils_plot import handle_chart_click as _handle_chart_click


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _leakage_for(source_df, usage_col, fee_col):
    """Estimate unmetered brand cost based on median rate × area."""
    met = source_df[source_df[usage_col] > 0]
    if len(met) < 2:
        return {}, 0.0
    med_rate = (met[fee_col] / met["size_m2"].replace(0, np.nan)).median()
    if not pd.notna(med_rate):
        return {}, 0.0
    unmet = source_df[source_df[usage_col] == 0]
    per_brand = {}
    for _, r in unmet.iterrows():
        per_brand[r["brand"]] = per_brand.get(r["brand"], 0) + float(r["size_m2"]) * med_rate
    total = sum(per_brand.values())
    return per_brand, total


def _grp_keys(df):
    """Return groupby keys: ['brand', 'building'] if building exists, else ['brand']."""
    return ["brand", "building"] if "building" in df.columns else ["brand"]


def _build_merged(
    water_df, hotwater_df, elec_df, billing_df, meter_df,
    prev_water_df=None, prev_hotwater_df=None, prev_elec_df=None, prev_billing_df=None,
):
    """Aggregate per-(brand, building) cross-sheet data and compute MoM columns.

    Returns (merged_df, has_prev).
    """
    # ── Per-sheet (brand, building) aggregation ────────────────────────────────
    def _gk(df):
        return ["brand", "building"] if "building" in df.columns else ["brand"]

    def _agg_sheet(df, agg_dict):
        if df is None or df.empty:
            return None
        gk = _gk(df)
        return df.groupby(gk).agg(**agg_dict).reset_index()

    _JOIN = ["brand", "building"]

    _w_agg = dict(floor=("floor", "first"), size_m2=("size_m2", "sum"),
                  water_total=("total", "sum"))
    if water_df is not None and "usage_m3" in (water_df.columns if water_df is not None else []):
        _w_agg["water_m3"] = ("usage_m3", "sum")
    _w = _agg_sheet(water_df, _w_agg)
    if _w is None:
        _w = pd.DataFrame(columns=["brand", "building", "floor", "size_m2", "water_total"])

    _hw_agg = dict(hw_total=("total", "sum"))
    if hotwater_df is not None and "usage_m3" in (hotwater_df.columns if hotwater_df is not None else []):
        _hw_agg["hw_m3"] = ("usage_m3", "sum")
    _hw = _agg_sheet(hotwater_df, _hw_agg)
    if _hw is None:
        _hw = pd.DataFrame(columns=_JOIN + ["hw_total"])

    _el_agg = dict(elec_total=("grand_total", "sum"), kwh_total=("kwh_total", "sum"))
    if elec_df is not None:
        if "kwh_ehp" in elec_df.columns:
            _el_agg["kwh_ehp"] = ("kwh_ehp", "sum")
        if "kwh_fcu" in elec_df.columns:
            _el_agg["kwh_fcu"] = ("kwh_fcu", "sum")
    _el = _agg_sheet(elec_df, _el_agg)
    if _el is not None:
        _el["kwh_hvac"] = _el.get("kwh_ehp", 0) + _el.get("kwh_fcu", 0)
        _el = _el.drop(columns=["kwh_ehp", "kwh_fcu"], errors="ignore")
    else:
        _el = pd.DataFrame(columns=_JOIN + ["elec_total", "kwh_total", "kwh_hvac"])

    if billing_df is not None and not billing_df.empty and "heat_total" in billing_df.columns:
        _ht = _agg_sheet(billing_df, dict(heat_total=("heat_total", "sum")))
        _ht["heat_total"] = _ht["heat_total"] * 10000  # 만원 → 원
    else:
        _ht = pd.DataFrame(columns=_JOIN + ["heat_total"])

    if meter_df is not None and not meter_df.empty and "heat_current" in meter_df.columns:
        from data import to_numeric_series as _tns
        _hm = meter_df.copy()
        _hm["heat_m3"] = _tns(_hm["heat_current"])
        _hm = _agg_sheet(_hm, dict(heat_m3=("heat_m3", "sum")))
    else:
        _hm = pd.DataFrame(columns=_JOIN + ["heat_m3"])

    # Ensure all frames have building column for join
    for _df in [_w, _hw, _el, _ht, _hm]:
        if "building" not in _df.columns:
            _df["building"] = ""

    merged = (_w.merge(_hw, on=_JOIN, how="outer")
                .merge(_el, on=_JOIN, how="outer")
                .merge(_ht, on=_JOIN, how="outer")
                .merge(_hm, on=_JOIN, how="outer"))

    for col in ["water_total", "hw_total", "elec_total", "heat_total", "kwh_total",
                "kwh_hvac", "size_m2", "water_m3", "hw_m3", "heat_m3"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)
        else:
            merged[col] = 0

    # Recover brand_raw for display
    _raw_parts = [
        df.groupby(_gk(df))["brand_raw"].first()
        for df in [water_df, hotwater_df, elec_df, billing_df, meter_df]
        if df is not None and not df.empty and "brand_raw" in df.columns
    ]
    if _raw_parts:
        _raw_map = pd.concat(_raw_parts)
        if isinstance(_raw_map.index, pd.MultiIndex):
            _raw_map = _raw_map.groupby(level=list(range(_raw_map.index.nlevels))).first()
            _raw_lookup = _raw_map.reset_index()
            _raw_lookup.columns = list(_raw_lookup.columns[:-1]) + ["brand_raw"]
            merged = merged.merge(_raw_lookup, on=_JOIN, how="left", suffixes=("", "_raw_dup"))
            merged.drop(columns=[c for c in merged.columns if c.endswith("_raw_dup")], inplace=True)
        else:
            _raw_map = _raw_map.groupby(level=0).first()
            merged["brand_raw"] = merged["brand"].map(_raw_map)

    # Fill missing building/floor/size from other sheets
    _meta_parts = [
        df.groupby(_gk(df))[["floor", "size_m2"]].first()
        for df in [water_df, hotwater_df, elec_df, billing_df]
        if df is not None and not df.empty
        and all(c in df.columns for c in ["floor", "size_m2"])
    ]
    if _meta_parts and "floor" in merged.columns:
        _meta = pd.concat(_meta_parts)
        if isinstance(_meta.index, pd.MultiIndex):
            _meta = _meta.groupby(level=list(range(_meta.index.nlevels))).first()
        else:
            _meta = _meta.groupby(level=0).first()
        # Only fill where floor is missing
        _no_floor = merged["floor"].isna() | (merged["floor"].astype(str).str.strip() == "")
        if isinstance(_meta.index, pd.MultiIndex):
            _meta_r = _meta.reset_index()
            _meta_r.columns = _JOIN + ["_fill_floor", "_fill_size"]
            merged = merged.merge(_meta_r, on=_JOIN, how="left")
            merged["floor"] = merged["floor"].where(~_no_floor, merged.get("_fill_floor"))
            merged["size_m2"] = merged["size_m2"].where(merged["size_m2"] > 0, merged.get("_fill_size", 0))
            merged.drop(columns=["_fill_floor", "_fill_size"], errors="ignore", inplace=True)

    merged["util_total"] = merged["water_total"] + merged["hw_total"] + merged["elec_total"] + merged["heat_total"]
    merged = merged[merged["util_total"] > 0].sort_values("util_total", ascending=False).reset_index(drop=True)

    from utils import display_brand as _display_brand
    merged = _display_brand(merged)

    # ── MoM ──────────────────────────────────────────────────────────────────
    _has_prev = any(d is not None and not d.empty
                    for d in [prev_water_df, prev_hotwater_df, prev_elec_df, prev_billing_df])
    if _has_prev:
        _prev_aggs = []
        if prev_water_df is not None and not prev_water_df.empty:
            _prev_aggs.append(("water_prev", prev_water_df, "total"))
        if prev_hotwater_df is not None and not prev_hotwater_df.empty:
            _prev_aggs.append(("hw_prev", prev_hotwater_df, "total"))
        if prev_elec_df is not None and not prev_elec_df.empty:
            _prev_aggs.append(("elec_prev", prev_elec_df, "grand_total"))
        if prev_billing_df is not None and not prev_billing_df.empty and "heat_total" in prev_billing_df.columns:
            _prev_aggs.append(("heat_prev", prev_billing_df, "heat_total"))

        for col_name, pdf, val_col in _prev_aggs:
            _pk = _gk(pdf)
            _prev_s = pdf.groupby(_pk)[val_col].sum()
            if col_name == "heat_prev":
                _prev_s = _prev_s * 10000
            if isinstance(_prev_s.index, pd.MultiIndex):
                _prev_r = _prev_s.reset_index()
                _prev_r.columns = _pk + [col_name]
                merged = merged.merge(_prev_r, on=_pk, how="left")
                merged[col_name] = merged[col_name].fillna(0)
            else:
                merged[col_name] = merged["brand"].map(_prev_s).fillna(0)

        for col in ["water_prev", "hw_prev", "elec_prev", "heat_prev"]:
            if col not in merged.columns:
                merged[col] = 0.0

        merged["util_prev"] = merged["water_prev"] + merged["hw_prev"] + merged["elec_prev"] + merged["heat_prev"]
        merged["util_change"] = merged["util_total"] - merged["util_prev"]

    return merged, _has_prev


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_mgmt_report(
    water_df, hotwater_df, elec_df, billing_df, meter_df,
    prev_water_df=None, prev_hotwater_df=None, prev_elec_df=None, prev_billing_df=None,
    split_by_building: bool = True,
) -> None:
    """Render the 📋 경영 보고 sub-tab inside 점검대상."""

    merged, _has_prev = _build_merged(
        water_df, hotwater_df, elec_df, billing_df, meter_df,
        prev_water_df, prev_hotwater_df, prev_elec_df, prev_billing_df,
    )

    if merged.empty:
        st.info("경영 보고에 필요한 유틸리티 데이터가 부족합니다.")
        return

    st.subheader("수익성 분석 — 경영 보고")
    st.caption("미청구 손실 추정, 이상 징후 브랜드, 비용 집중도를 종합한 경영용 보고서입니다.")

    # ── Leakage computation per sheet ─────────────────────────────────────────
    _w_per_brand,  _w_total_leak  = _leakage_for(water_df,    "usage_m3",  "total")    if water_df    is not None else ({}, 0.0)
    _hw_per_brand, _hw_total_leak = _leakage_for(hotwater_df, "usage_m3",  "total")    if hotwater_df is not None else ({}, 0.0)
    _el_per_brand, _el_total_leak = _leakage_for(elec_df,     "kwh_total", "grand_total") if elec_df is not None else ({}, 0.0)

    # ── IQR anomaly count ─────────────────────────────────────────────────────
    _util_up = _iqr_upper(merged["util_total"])
    _n_anomaly = int((merged["util_total"] > _util_up).sum())

    # ── Top-5 cost concentration ──────────────────────────────────────────────
    _total_spend = merged["util_total"].sum()
    _mgmt_top5 = merged.nlargest(5, "util_total")
    _top5_pct = _mgmt_top5["util_total"].sum() / _total_spend * 100 if _total_spend else 0

    # ── Unmetered summary ─────────────────────────────────────────────────────
    _all_unmet_brands = set(_w_per_brand) | set(_hw_per_brand) | set(_el_per_brand)
    _all3_unmet = set(_w_per_brand) & set(_hw_per_brand) & set(_el_per_brand)

    # ── Data insight expander ─────────────────────────────────────────────────
    _mgmt_top1_brand = merged.loc[merged["util_total"].idxmax(), "brand"]
    _mgmt_top1_val   = merged["util_total"].max()
    _mgmt_top5_share = _top5_pct
    with st.expander("현재 데이터 해석"):
        st.markdown(f"""
#### 즉각 조치
- **고비용 이상치**: {"없음 — 현재 정상 범위 내 ✅" if _n_anomaly == 0 else f"⚠️ **{_n_anomaly}개** 브랜드가 IQR 상한({_util_up/1e4:,.0f}만 원) 초과 → 누수·설비 과부하·계량 오류 여부 점검"}
- **전 항목 미계량**: {"없음 ✅" if not _all3_unmet else f"⚠️ **{len(_all3_unmet)}개** 브랜드 수도·온수·전기 모두 미계량 ({', '.join(sorted(_all3_unmet)[:5])}{'…' if len(_all3_unmet)>5 else ''}) → 계약 및 미터 설치 즉시 확인"}

#### 비용 집중도 & 리스크
- 상위 5개 브랜드가 전체의 **{_mgmt_top5_share:.1f}%** 차지 ({', '.join(_mgmt_top5['brand'].astype(str).str[:10].tolist())})
{"- ⚠️ 집중도 과다 — 해당 브랜드 이탈 시 수익 구조에 큰 영향. 계약 조건 재검토 권장" if _mgmt_top5_share > 50 else "- ✅ 비용 집중도 분산 — 특정 브랜드 의존도 낮음"}
- **최고 비용 브랜드**: {_mgmt_top1_brand} — {_mgmt_top1_val/1e4:,.0f}만 원 (전체의 **{_mgmt_top1_val/_total_spend*100:.1f}%**)

#### 운영 효율
- 전체 브랜드 수: {len(merged)}개 · 총 유틸리티 지출: {_fmt_won(_total_spend)}
- 브랜드당 평균: {_fmt_won(_total_spend/len(merged))} · 미계량 포함 브랜드: {len(_all_unmet_brands)}개
""")

    # ── Cross-tab synthesis ───────────────────────────────────────────────────
    _es_util_total  = merged["util_total"].sum()
    _es_n           = len(merged)
    _es_avg         = _es_util_total / _es_n
    _es_med         = merged["util_total"].median()
    _es_top3_share  = merged.nlargest(3, "util_total")["util_total"].sum() / _es_util_total * 100
    _es_top10p_n    = max(1, int(_es_n * 0.1))
    _es_top10p_share = merged.nlargest(_es_top10p_n, "util_total")["util_total"].sum() / _es_util_total * 100
    _es_outlier_burden = merged[merged["util_total"] > _util_up]["util_total"].sum() / _es_util_total * 100 if _n_anomaly > 0 else 0.0
    _es_skew        = _es_avg > _es_med * 1.1

    # Area efficiency
    _es_adf = merged[merged["size_m2"] > 0].copy()
    _es_adf["pm2"] = _es_adf["util_total"] / _es_adf["size_m2"]
    _es_a_iqr_up   = _iqr_upper(_es_adf["pm2"])
    _es_n_area_over = int((_es_adf["pm2"] > _es_a_iqr_up).sum())
    _es_area_top1  = _es_adf.loc[_es_adf["pm2"].idxmax()] if not _es_adf.empty else None

    # Building
    _es_bld = merged.groupby("building").agg(
        util=("util_total", "sum"), cnt=("brand", "count"), area=("size_m2", "sum")
    ).reindex(["A", "B", "C", "D"]).dropna(how="all")
    _es_bld["pm2"] = _es_bld["util"] / _es_bld["area"].replace(0, float("nan"))
    _es_bld_max    = _es_bld["util"].idxmax() if not _es_bld.empty else "-"
    _es_bld_eff    = _es_bld["pm2"].idxmin() if _es_bld["pm2"].notna().any() else "-"
    _es_bld_ratio  = _es_bld["util"].max() / _es_bld["util"].min() if _es_bld["util"].min() > 0 else 1

    # Utility mix
    _es_w_pct  = merged["water_total"].sum() / _es_util_total * 100
    _es_hw_pct = merged["hw_total"].sum()    / _es_util_total * 100
    _es_el_pct = merged["elec_total"].sum()  / _es_util_total * 100
    _es_ht_pct = merged["heat_total"].sum()  / _es_util_total * 100
    _es_dom    = max([("수도", _es_w_pct), ("온수", _es_hw_pct), ("전기", _es_el_pct), ("난방", _es_ht_pct)], key=lambda x: x[1])

    # Build prioritized findings
    _findings, _actions, _strategy = [], [], []

    if _n_anomaly > 0:
        _findings.append(f"고비용 이상치 **{_n_anomaly}개** 브랜드가 전체 유틸리티 비용의 **{_es_outlier_burden:.1f}%** 를 점유합니다.")
        _actions.append(f"🔴 **즉시** — 이상치 {_n_anomaly}개 브랜드 계량기·누수·설비 과부하 현장 점검")
    if _all3_unmet:
        _findings.append(f"수도·온수·전기 **전부** 미계량 브랜드가 **{len(_all3_unmet)}개** 있습니다. 미터 미설치 또는 계약 누락 가능성이 있습니다.")
        _actions.append(f"🔴 **즉시** — 전 항목 미계량 {len(_all3_unmet)}개 브랜드 계약서·미터 설치 현황 확인")
    if _es_n_area_over > 0:
        _top_pm2_brand = _es_area_top1["brand"] if _es_area_top1 is not None else "-"
        _findings.append(f"면적당 비용 이상 고점 **{_es_n_area_over}개** 브랜드 — 대표: {_top_pm2_brand} ({_es_area_top1['pm2']:,.0f} 원/㎡). 동일 면적 대비 과소비 수준입니다.")
        _actions.append(f"🟠 **검토** — 면적당 비용 상위 이상치 {_es_n_area_over}개 브랜드 사용 패턴 및 설비 상태 점검")
    if len(_all_unmet_brands) > 0:
        _findings.append(f"부분 미계량(수도·온수·전기 중 일부 누락) 브랜드가 **{len(_all_unmet_brands)}개** 있습니다. 업종 특성일 수 있으나 확인이 필요합니다.")
        _actions.append(f"🟡 **관찰** — 부분 미계량 {len(_all_unmet_brands)}개 브랜드 업종 특성 검토 후 계량 불필요 여부 판단")
    if _es_top10p_share > 50:
        _findings.append(f"상위 10% ({_es_top10p_n}개 브랜드)가 전체 비용의 **{_es_top10p_share:.1f}%** 를 차지합니다. 소수 브랜드 의존도가 높습니다.")
        _strategy.append("비용 집중 브랜드의 임대 계약 갱신 조건 재검토 — 이탈 시 수익 구조 충격 최소화 방안 마련")
    if _es_el_pct > 50:
        _findings.append(f"전기 비중이 **{_es_el_pct:.1f}%** 로 절반 이상을 차지합니다. 냉난방 설비 효율이 핵심 관리 포인트입니다.")
        _strategy.append("전력 다소비 구간 피크 저감 프로그램 도입 또는 LED·인버터 설비 교체 검토")
    if _es_skew:
        _findings.append(f"평균({_fmt_won(_es_avg)})이 중앙값({_fmt_won(_es_med)})보다 높아 소수 고비용 브랜드가 전체 평균을 왜곡하고 있습니다.")
    if _es_bld_ratio > 2.5:
        _findings.append(f"건물별 비용 격차가 크며({_es_bld_max}동 최고), **{_es_bld_eff}동** 이 면적당 비용 최저로 효율 벤치마크가 됩니다.")
        _strategy.append(f"{_es_bld_eff}동 운영 방식을 타 건물에 벤치마킹하여 면적당 비용 절감 목표 수립 권장")

    if not _findings:
        _findings.append("현재 데이터 기준으로 즉각 조치가 필요한 이상 징후는 발견되지 않았습니다.")

    st.markdown("### 📋 종합 경영 요약")
    st.markdown(f"분석 대상 **{_es_n}개 브랜드** · 총 유틸리티 지출 **{_fmt_won(_es_util_total)}** · 지배 항목 **{_es_dom[0]} ({_es_dom[1]:.0f}%)**")
    st.divider()

    fc1, fc2 = st.columns([1, 1])
    with fc1:
        st.markdown("#### 🔍 주요 발견사항")
        for i, f in enumerate(_findings, 1):
            st.markdown(f"{i}. {f}")

    with fc2:
        st.markdown("#### ✅ 조치 항목 (우선순위순)")
        if _actions:
            for a in _actions:
                st.markdown(f"- {a}")
        else:
            st.markdown("- 즉각 조치 항목 없음")
        if _strategy:
            st.markdown("#### 💡 전략적 제언")
            for s in _strategy:
                st.markdown(f"- {s}")

    st.divider()

    # ── KPI row ───────────────────────────────────────────────────────────────
    kc = st.columns(4)
    kc[0].metric("미계량 브랜드 (검토 필요)",
                 f"{len(_all_unmet_brands)}개",
                 help="수도/온수/전기 중 하나 이상 미계량. 업종 특성상 정상일 수 있습니다.")
    kc[1].metric("전 유틸리티 미계량",
                 f"{len(_all3_unmet)}개",
                 help="수도·온수·전기 모두 미계량 — 계약·미터 점검 권장")
    kc[2].metric("이상 징후 브랜드",
                 f"{_n_anomaly}개",
                 help="총 유틸리티 비용 IQR 상한 초과")
    kc[3].metric("상위 5개 비중",
                 f"{_top5_pct:.1f}%",
                 help="전체 유틸리티 지출에서 상위 5개 브랜드 비중")

    if _all3_unmet:
        st.markdown(f"#### 전체 미계량 브랜드 ({len(_all3_unmet)}개) — 계약·미터 즉시 확인")
        _unmet_rows = []
        for b in sorted(_all3_unmet):
            _row = merged[merged["brand"] == b]
            _bld  = _row["building"].iloc[0] if not _row.empty else "-"
            _flr  = str(_row["floor"].iloc[0]) if not _row.empty else "-"
            _util = int(_row["util_total"].iloc[0]) if not _row.empty else 0
            _unmet_rows.append({
                "브랜드":        b,
                "건물":          _bld,
                "층":            _flr,
                "총비용 (만원)": round(_util / 1e4, 1),
            })
        st.dataframe(
            pd.DataFrame(_unmet_rows),
            column_config={
                "브랜드":        st.column_config.TextColumn("브랜드"),
                "건물":          st.column_config.TextColumn("건물", width="small"),
                "층":            st.column_config.TextColumn("층",   width="small"),
                "총비용 (만원)": st.column_config.NumberColumn("총비용 (만원)", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── 공실 이상 소비 분석 ───────────────────────────────────────────────────
    _vacancy_df = merged[merged["brand"].astype(str).str.contains("공실", na=False)].copy()
    if not _vacancy_df.empty:
        st.markdown(f"#### 🏚 공실 유틸리티 이상 소비 분석 ({len(_vacancy_df)}개 공실)")
        st.caption("🔴 고의심: 중앙값 30%↑ · 🟠 주의: 소량 소비 · 🟢 정상: 소비 없음")
        _global_med = merged["util_total"].median()
        _vac_rows = []
        for _, vrow in _vacancy_df.iterrows():
            _w  = float(vrow.get("water_total", 0) or 0)
            _hw = float(vrow.get("hw_total",    0) or 0)
            _el = float(vrow.get("elec_total",  0) or 0)
            _ht = float(vrow.get("heat_total",  0) or 0)
            _tot = float(vrow["util_total"])
            if _tot > _global_med * 0.3:
                _suspicion = "🔴 고의심 — 무단점거·대형 누수 가능성"
            elif _tot > 0:
                _suspicion = "🟠 주의 — 소량 소비, 누수·계량기 오류 확인 필요"
            else:
                _suspicion = "🟢 정상 — 소비 없음"
            _vac_rows.append({
                "공실":          str(vrow["brand"]),
                "건물":          str(vrow.get("building", "-")),
                "층":            str(vrow.get("floor", "-")),
                "수도 (만원)":   round(_w  / 1e4, 1),
                "온수 (만원)":   round(_hw / 1e4, 1),
                "전기 (만원)":   round(_el / 1e4, 1),
                "난방 (만원)":   round(_ht / 1e4, 1),
                "합계 (만원)":   round(_tot / 1e4, 1),
                "의심 수준":     _suspicion,
            })
        _vac_df = pd.DataFrame(_vac_rows).sort_values("합계 (만원)", ascending=False)
        _vac_consuming = (_vac_df["합계 (만원)"] > 0).sum()
        if _vac_consuming > 0:
            st.warning(f"⚠️ {_vac_consuming}개 공실에서 유틸리티 소비가 감지되었습니다. 무단점거·누수·계량기 오류 여부를 즉시 확인하세요.")
        else:
            st.success("✅ 모든 공실의 유틸리티 소비가 0입니다.")
        st.dataframe(
            _vac_df,
            column_config={
                "공실":        st.column_config.TextColumn("공실"),
                "건물":        st.column_config.TextColumn("건물", width="small"),
                "층":          st.column_config.TextColumn("층",   width="small"),
                "수도 (만원)": st.column_config.NumberColumn("수도 (만원)",  format="%.1f"),
                "온수 (만원)": st.column_config.NumberColumn("온수 (만원)",  format="%.1f"),
                "전기 (만원)": st.column_config.NumberColumn("전기 (만원)",  format="%.1f"),
                "난방 (만원)": st.column_config.NumberColumn("난방 (만원)",  format="%.1f"),
                "합계 (만원)": st.column_config.NumberColumn("합계 (만원)",  format="%.1f"),
                "의심 수준":   st.column_config.TextColumn("의심 수준"),
            },
            use_container_width=True,
            hide_index=True,
        )
        st.divider()

    # ── Priority action table ─────────────────────────────────────────────────
    st.markdown("#### 우선 조치 브랜드")
    st.caption("🔴 즉시(≥4점) · 🟠 검토(2–3점) · 🟡 관찰(1점) · 🟢 정상(0점) — "
               "이상치 ×3 + 전체미계량 ×2 + 저납부 ×1")

    # Precompute building medians (avoid N+1 recomputation per row)
    _bld_med_cache = (
        (merged["util_total"] / merged["size_m2"].replace(0, np.nan))
        .groupby(merged["building"]).median()
    )

    _action_rows = []
    for _, row in merged.iterrows():
        b = str(row["brand"])
        _is_anom = bool(row["util_total"] > _util_up)
        _w_u  = b in _w_per_brand
        _hw_u = b in _hw_per_brand
        _el_u = b in _el_per_brand
        _all3 = _w_u and _hw_u and _el_u
        _pm2 = row["util_total"] / row["size_m2"] if row.get("size_m2", 0) > 0 else np.nan
        _score = (_is_anom * 3) + (_all3 * 2)

        _bld = str(row.get("building", ""))
        _bld_med_pm2 = _bld_med_cache.get(_bld, np.nan)
        _underpay = (pd.notna(_pm2) and pd.notna(_bld_med_pm2) and _bld_med_pm2 > 0
                     and _pm2 < _bld_med_pm2 * 0.4)
        if _underpay:
            _score += 1

        _reasons = []
        if _is_anom:
            _mult = row["util_total"] / _util_up if _util_up > 0 else 0
            _reasons.append(f"총비용이 IQR 상한의 {_mult:.1f}배 — 누수·설비 과부하·계량 오류 의심")
        if _all3:
            _reasons.append("수도·온수·전기 전부 미계량 — 계약 누락 또는 미터 미설치 가능성")
        elif _w_u or _hw_u or _el_u:
            _items = ("수도" if _w_u else "") + ("·온수" if _hw_u else "") + ("·전기" if _el_u else "")
            _reasons.append(f"{_items.strip('·')} 미계량 — 업종 특성 또는 미터 오류 확인 필요")
        if _underpay and pd.notna(_bld_med_pm2):
            _ratio = _pm2 / _bld_med_pm2 * 100 if _bld_med_pm2 > 0 else 0
            _reasons.append(f"동 건물 중앙값의 {_ratio:.0f}% 수준 — 미계량·장기 휴업 가능성")

        _action_rows.append({
            "브랜드":        b,
            "건물":          _bld,
            "층":            str(row.get("floor", "")),
            "총비용 (만원)": round(row["util_total"] / 1e4, 1),
            "원/m²":         round(_pm2, 0) if pd.notna(_pm2) else None,
            "건물중앙 원/m²": round(_bld_med_pm2, 0) if pd.notna(_bld_med_pm2) else None,
            "주의 사유":     " / ".join(_reasons) if _reasons else "-",
            "우선순위 점수": _score,
        })

    _action_df = (
        pd.DataFrame(_action_rows)
        .sort_values(["우선순위 점수", "총비용 (만원)"], ascending=[False, False])
        .reset_index(drop=True)
    )
    _action_df.index = _action_df.index + 1
    _action_df.insert(0, "등급", _action_df["우선순위 점수"].map(
        lambda s: "🔴 즉시" if s >= 4 else ("🟠 검토" if s >= 2 else ("🟡 관찰" if s == 1 else "🟢 정상"))
    ))

    st.dataframe(
        _action_df,
        column_config={
            "등급":           st.column_config.TextColumn("등급", width="small"),
            "브랜드":         st.column_config.TextColumn("브랜드"),
            "건물":           st.column_config.TextColumn("건물", width="small"),
            "층":             st.column_config.TextColumn("층", width="small"),
            "총비용 (만원)":  st.column_config.NumberColumn("총비용 (만원)", format="%.1f"),
            "원/m²":          st.column_config.NumberColumn("원/m²", format="%,.0f"),
            "건물중앙 원/m²": st.column_config.NumberColumn("건물중앙 원/m²", format="%,.0f"),
            "주의 사유":      st.column_config.TextColumn("주의 사유", width="large"),
            "우선순위 점수":  st.column_config.ProgressColumn("우선순위", format="%d", min_value=0, max_value=6),
        },
        use_container_width=True,
    )

    # ── Quick brand → profile buttons for top priority brands ────────────────
    _priority_brands = _action_df[_action_df["우선순위 점수"] > 0]["브랜드"].tolist()[:10]
    if _priority_brands:
        _pbcols = st.columns(min(len(_priority_brands), 5))
        for _pi, _pb in enumerate(_priority_brands):
            with _pbcols[_pi % min(len(_priority_brands), 5)]:
                if st.button(f"🔍 {_pb}", key=f"mgmt_action_goto_{_pi}",
                             use_container_width=True):
                    st.session_state["_goto_profile_brand"] = _pb
                    st.rerun()

    # ── Download management report ────────────────────────────────────────────
    _buf = io.BytesIO()
    with pd.ExcelWriter(_buf, engine="openpyxl") as _xw:
        _action_df.to_excel(_xw, sheet_name="우선조치목록", index=True)
        for _sheet_label, _per_brand in [("수도_미계량", _w_per_brand),
                                          ("온수_미계량", _hw_per_brand),
                                          ("전기_미계량", _el_per_brand)]:
            if _per_brand:
                _detail = pd.DataFrame([{"브랜드": k} for k in _per_brand])
                _detail.to_excel(_xw, sheet_name=_sheet_label, index=False)
    st.download_button(
        "📥 경영 보고서 다운로드 (Excel)",
        data=_buf.getvalue(),
        file_name="utility_management_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="mgmt_dl_btn",
    )

    st.divider()

    # ── Cost concentration (Pareto) ───────────────────────────────────────────
    st.markdown("#### 비용 집중도 (Pareto)")
    _sorted = merged.sort_values("util_total", ascending=False).reset_index(drop=True)
    _sorted["누적 비중 (%)"] = (_sorted["util_total"].cumsum() / _total_spend * 100).round(1)
    _sorted["순위"] = _sorted.index + 1

    _pc1, _pc2 = st.columns([3, 1])
    with _pc1:
        _pareto_n = st.slider("표시 브랜드 수", 5, len(_sorted), min(20, len(_sorted)), key="mgmt_pareto_n")
    with _pc2:
        _pareto_logy = st.checkbox("Log 스케일", key="mgmt_pareto_logy")
    _pareto = _sorted.head(_pareto_n)

    fig_p = go.Figure()
    fig_p.add_trace(go.Bar(
        x=[str(b)[:22] for b in _pareto["brand"]],
        y=_pareto["util_total"].values,
        marker_color=(
            [_BLD_COLOR.get(str(b), "#888") for b in _pareto["building"]]
            if split_by_building else "#4C72B0"
        ),
        text=[f"{v/1e6:.2f}M" for v in _pareto["util_total"].values],
        textposition="outside", textfont=dict(size=9),
        name="유틸리티 합계",
    ))
    fig_p.add_trace(go.Scatter(
        x=[str(b)[:22] for b in _pareto["brand"]],
        y=_pareto["누적 비중 (%)"].values,
        mode="lines+markers", name="누적 비중 (%)",
        yaxis="y2", line=dict(color="#C44E52", width=2),
        marker=dict(size=5),
    ))
    _y1_cfg = dict(title="원", gridcolor="rgba(128,128,128,0.2)", griddash="dot")
    if _pareto_logy:
        _y1_cfg["type"] = "log"
    fig_p.update_layout(
        height=400, plot_bgcolor="rgba(0,0,0,0)",
        yaxis=_y1_cfg,
        yaxis2=dict(title="누적 비중 (%)", overlaying="y", side="right",
                    range=[0, 105], showgrid=False),
        xaxis=dict(tickangle=-40),
        legend=dict(orientation="h", x=1, y=1, xanchor="right", yanchor="top"),
        margin=dict(l=10, r=60, t=50, b=100),
    )
    _ev_pareto = st.plotly_chart(fig_p, use_container_width=True, key="mgmt_pareto_chart", on_select="rerun")
    _handle_chart_click(_ev_pareto, _pareto, brand_col="brand", field="x")

    _p80 = int((_sorted["누적 비중 (%)"] <= 80).sum()) + 1
    st.caption(
        f"상위 **{_p80}개** 브랜드가 전체 비용의 80%를 차지합니다. "
        f"(전체 {len(merged)}개 브랜드 중 {_p80/len(merged)*100:.0f}%)"
    )

    # ── MoM 비용 변화 요약 (Top Movers) ─────────────────────────────────────
    if _has_prev and "util_change" in merged.columns:
        st.divider()
        st.markdown("#### 📈 전월 대비 비용 변화 — Top Movers")
        _mom_total_chg = merged["util_change"].sum()
        _mom_total_pct = _mom_total_chg / merged["util_prev"].sum() * 100 if merged["util_prev"].sum() > 0 else 0
        st.caption(
            f"전체 비용 변화: **{_fmt_won(abs(_mom_total_chg))}** "
            f"({'증가' if _mom_total_chg > 0 else '감소'}, {_mom_total_pct:+.1f}%)"
        )

        _top_inc = merged.nlargest(5, "util_change")[["brand", "building", "util_prev", "util_total", "util_change"]].copy()
        _top_dec = merged.nsmallest(5, "util_change")[["brand", "building", "util_prev", "util_total", "util_change"]].copy()

        _mc1, _mc2 = st.columns(2)
        with _mc1:
            st.markdown("**🔺 비용 증가 Top 5**")
            if not _top_inc.empty and _top_inc["util_change"].max() > 0:
                _ti = _top_inc[_top_inc["util_change"] > 0].copy()
                _ti["전월"] = _ti["util_prev"].apply(_fmt_won)
                _ti["이번달"] = _ti["util_total"].apply(_fmt_won)
                _ti["변화"] = _ti["util_change"].apply(lambda v: _fmt_won(v, signed=True))
                _ti["변화율(%)"] = (_ti["util_change"] / _ti["util_prev"].replace(0, np.nan) * 100).round(1)
                _ti_disp = _ti[["brand", "building", "전월", "이번달", "변화", "변화율(%)"]].rename(
                    columns={"brand": "브랜드", "building": "건물"})
                st.dataframe(_ti_disp.reset_index(drop=True), hide_index=True, use_container_width=True)

                _top1_inc = _ti.iloc[0]
                _inc_pct = _top1_inc["util_change"] / _top1_inc["util_prev"] * 100 if _top1_inc["util_prev"] > 0 else 0
                if _inc_pct > 50:
                    st.warning(
                        f"⚠️ **{_top1_inc['brand']}** 비용이 전월 대비 {_inc_pct:.0f}% 급등 — "
                        f"누수·장비 이상·영업 확장 여부를 확인하세요"
                    )
                elif _inc_pct > 20:
                    st.info(
                        f"📌 **{_top1_inc['brand']}** 비용이 {_inc_pct:.0f}% 증가 — "
                        f"계절 패턴인지 사용 행태 변화인지 모니터링이 필요합니다"
                    )
            else:
                st.success("비용 증가 브랜드 없음")

        with _mc2:
            st.markdown("**🔻 비용 감소 Top 5**")
            if not _top_dec.empty and _top_dec["util_change"].min() < 0:
                _td = _top_dec[_top_dec["util_change"] < 0].copy()
                _td["전월"] = _td["util_prev"].apply(_fmt_won)
                _td["이번달"] = _td["util_total"].apply(_fmt_won)
                _td["변화"] = _td["util_change"].apply(lambda v: _fmt_won(v, signed=True))
                _td["변화율(%)"] = (_td["util_change"] / _td["util_prev"].replace(0, np.nan) * 100).round(1)
                _td_disp = _td[["brand", "building", "전월", "이번달", "변화", "변화율(%)"]].rename(
                    columns={"brand": "브랜드", "building": "건물"})
                st.dataframe(_td_disp.reset_index(drop=True), hide_index=True, use_container_width=True)

                _top1_dec = _td.iloc[0]
                _dec_pct = abs(_top1_dec["util_change"]) / _top1_dec["util_prev"] * 100 if _top1_dec["util_prev"] > 0 else 0
                if _dec_pct > 50:
                    st.info(
                        f"📌 **{_top1_dec['brand']}** 비용이 {_dec_pct:.0f}% 급감 — "
                        f"공실 전환·영업 축소·계량 오류 가능성을 확인하세요"
                    )
            else:
                st.caption("비용 감소 브랜드 없음")

        # MoM waterfall chart — top 10 movers
        _movers = merged[merged["util_change"].abs() > 0].nlargest(10, "util_change", keep="all")
        _movers_bot = merged[merged["util_change"] < 0].nsmallest(5, "util_change")
        _mover_df = pd.concat([_movers, _movers_bot]).drop_duplicates(subset="brand").sort_values("util_change", ascending=False)
        if len(_mover_df) >= 3:
            _mom_wf_logy = st.checkbox("Log 스케일", key="mgmt_mom_wf_logy")
            _m_colors = _mover_df["util_change"].apply(
                lambda v: "#C44E52" if v > 0 else "#2ca02c"
            ).tolist()
            fig_mom = go.Figure(go.Bar(
                x=[str(b)[:20] for b in _mover_df["brand"]],
                y=_mover_df["util_change"],
                marker_color=_m_colors,
                text=_mover_df["util_change"].apply(lambda v: _fmt_won(v, signed=True)),
                textposition="outside",
                textfont=dict(size=9),
            ))
            fig_mom.add_hline(y=0, line_color="#888", line_width=1)
            fig_mom.update_layout(
                title="전월 대비 비용 변화 — 주요 변동 브랜드",
                height=380, xaxis_tickangle=-40,
                yaxis_title="변화 (원)",
                yaxis_type="log" if _mom_wf_logy else None,
                margin=dict(t=55, b=80, l=60, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_mom, use_container_width=True, key="mgmt_mom_waterfall")

    # ── 건물별 효율 벤치마크 ──────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 🏢 건물별 효율 벤치마크")
    _bld_bench = merged[merged["size_m2"] > 0].copy()
    if not _bld_bench.empty and "building" in _bld_bench.columns:
        _bld_agg = _bld_bench.groupby("building").agg(
            총비용=("util_total", "sum"),
            면적합계=("size_m2", "sum"),
            브랜드수=("brand", "count"),
            수도=("water_total", "sum"),
            온수=("hw_total", "sum"),
            전기=("elec_total", "sum"),
            난방=("heat_total", "sum"),
        ).reset_index()
        _bld_agg["원_per_m2"] = (_bld_agg["총비용"] / _bld_agg["면적합계"]).round(0)
        _bld_agg = _bld_agg.sort_values("원_per_m2", ascending=False)

        _mgmt_bld_logy = st.checkbox("Log 스케일", key="mgmt_bld_bench_logy")
        _bc1, _bc2 = st.columns([3, 2])
        with _bc1:
            _bld_stack = _bld_agg.copy()
            for _uc, _ul in [("수도", "💧 수도"), ("온수", "🌡 온수"),
                              ("전기", "⚡ 전기"), ("난방", "🔥 난방")]:
                _bld_stack[f"{_uc}_pm2"] = (_bld_stack[_uc] / _bld_stack["면적합계"]).round(0)

            fig_bld = go.Figure()
            for _uc, _ul, _clr in [("수도", "수도", "#4C72B0"), ("온수", "온수", "#C44E52"),
                                    ("전기", "전기", "#DD8A00"), ("난방", "난방", "#E377C2")]:
                fig_bld.add_trace(go.Bar(
                    name=_ul,
                    x=_bld_stack["building"],
                    y=_bld_stack[f"{_uc}_pm2"],
                    marker_color=_clr,
                    text=_bld_stack[f"{_uc}_pm2"].apply(lambda v: f"{v:,.0f}"),
                    textposition="inside",
                    textfont=dict(size=9, color="white"),
                ))
            fig_bld.update_layout(
                barmode="stack",
                title="건물별 면적당 유틸리티 비용 (원/m²)",
                height=380,
                yaxis_title="원/m²",
                yaxis_type="log" if _mgmt_bld_logy else None,
                margin=dict(t=55, b=40, l=60, r=20),
                legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center"),
            )
            st.plotly_chart(fig_bld, use_container_width=True, key="mgmt_bld_bench")

        with _bc2:
            _best_bld = _bld_agg.iloc[-1]
            _worst_bld = _bld_agg.iloc[0]
            _ratio = _worst_bld["원_per_m2"] / _best_bld["원_per_m2"] if _best_bld["원_per_m2"] > 0 else 1

            st.markdown("**📊 건물별 비용 효율**")
            _bld_disp = _bld_agg[["building", "브랜드수", "면적합계", "총비용", "원_per_m2"]].copy()
            _bld_disp["총비용"] = _bld_disp["총비용"].apply(_fmt_won)
            _bld_disp["면적합계"] = _bld_disp["면적합계"].apply(lambda v: f"{v:,.0f} m²")
            _bld_disp["원_per_m2"] = _bld_disp["원_per_m2"].apply(lambda v: f"{v:,.0f}")
            _bld_disp = _bld_disp.rename(columns={
                "building": "건물", "원_per_m2": "원/m²",
            })
            st.dataframe(_bld_disp, hide_index=True, use_container_width=True)

            if _ratio >= 1.5:
                st.warning(
                    f"⚠️ **{_worst_bld['building']}동**이 **{_best_bld['building']}동** 대비 "
                    f"면적당 비용 **{_ratio:.1f}배** → "
                    f"{_best_bld['building']}동의 운영 방식을 벤치마킹하여 절감 목표 수립 권장"
                )
            else:
                st.success("건물 간 면적당 비용 격차가 적습니다 (효율 편차 낮음)")

        # MoM building change
        if _has_prev and "util_change" in merged.columns:
            _bld_mom = merged.groupby("building").agg(
                이번달=("util_total", "sum"),
                전월=("util_prev", "sum"),
                변화=("util_change", "sum"),
            ).reset_index()
            _bld_mom["변화율(%)"] = (_bld_mom["변화"] / _bld_mom["전월"].replace(0, np.nan) * 100).round(1)
            _bld_mom_disp = _bld_mom.copy()
            _bld_mom_disp["이번달"] = _bld_mom_disp["이번달"].apply(_fmt_won)
            _bld_mom_disp["전월"] = _bld_mom_disp["전월"].apply(_fmt_won)
            _bld_mom_disp["변화"] = _bld_mom_disp["변화"].apply(lambda v: _fmt_won(v, signed=True))
            _bld_mom_disp = _bld_mom_disp.rename(columns={"building": "건물"})
            st.markdown("**📈 건물별 전월 대비 변화**")
            st.dataframe(_bld_mom_disp, hide_index=True, use_container_width=True)

    # ── 관심 브랜드 상태 요약 ─────────────────────────────────────────────────
    _watchlist = st.session_state.get("_brand_watchlist", [])
    if _watchlist:
        st.divider()
        st.markdown("#### ⭐ 관심 브랜드 상태")
        st.caption("이상감지에서 등록된 관심 브랜드의 현재 비용 현황입니다.")
        _wl_rows = []
        for _wb in _watchlist:
            _wr = merged[merged["brand"] == _wb]
            if _wr.empty:
                continue
            _wr = _wr.iloc[0]
            _wl_entry = {
                "브랜드": _wb,
                "건물": str(_wr.get("building", "—")),
                "총비용": _fmt_won(_wr["util_total"]),
            }
            if _has_prev and "util_change" in merged.columns:
                _wl_chg = _wr.get("util_change", 0)
                _wl_prev = _wr.get("util_prev", 0)
                _wl_pct = _wl_chg / _wl_prev * 100 if _wl_prev > 0 else 0
                _wl_entry["변화"] = _fmt_won(_wl_chg, signed=True)
                _wl_entry["변화율"] = f"{_wl_pct:+.1f}%"
                if abs(_wl_pct) >= 50:
                    _wl_entry["상태"] = "🔴 급변"
                elif abs(_wl_pct) >= 20:
                    _wl_entry["상태"] = "🟠 주의"
                else:
                    _wl_entry["상태"] = "🟢 안정"
            else:
                _rank = int((merged["util_total"] > _wr["util_total"]).sum()) + 1
                _pctile = _rank / len(merged) * 100
                _wl_entry["순위"] = f"{_rank}위 / {len(merged)}"
                _wl_entry["상태"] = "🔴 상위" if _pctile <= 10 else "🟠 중상위" if _pctile <= 30 else "🟢 보통"

            _wl_rows.append(_wl_entry)

        if _wl_rows:
            _wl_df = pd.DataFrame(_wl_rows)
            st.dataframe(_wl_df, hide_index=True, use_container_width=True)

            _wl_cols = st.columns(min(len(_wl_rows), 5))
            for _wi, _wr in enumerate(_wl_rows):
                with _wl_cols[_wi % min(len(_wl_rows), 5)]:
                    if st.button(f"🔍 {_wr['브랜드']}", key=f"mgmt_goto_{_wi}",
                                 use_container_width=True):
                        st.session_state["_goto_profile_brand"] = _wr["브랜드"]
                        st.rerun()

            if _has_prev:
                _wl_alarming = [r for r in _wl_rows if "상태" in r and "🔴" in r.get("상태", "")]
                if _wl_alarming:
                    st.warning(
                        f"⚠️ 관심 브랜드 중 **{len(_wl_alarming)}개**에서 급격한 변화 감지: "
                        + ", ".join(f"**{r['브랜드']}**" for r in _wl_alarming)
                        + " → 즉시 확인이 필요합니다"
                    )
        else:
            st.caption("관심 브랜드가 현재 데이터에 없습니다.")
