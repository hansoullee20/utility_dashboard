"""electricity.py — 전체 전기 사용내역 analysis view."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import BLD_COLOR as _BLD_COLOR, iqr_upper as _iqr_upper, flag_prefix as _flag_prefix

_USAGE_COLS = [
    ("전기01 (검침)",  "kwh_elec01",      "#4C72B0"),
    ("전기02 (검침)",  "kwh_elec02",      "#6A9CC8"),
    ("EHP",           "kwh_ehp",         "#C44E52"),
    ("주방배기펜",     "kwh_kitchen_fan", "#DD8A00"),
    ("중앙공조 AHU",  "kwh_ahu",         "#8172B2"),
    ("FCU 기계실",    "kwh_fcu",         "#55A868"),
    ("주방배수펌프",   "kwh_pump",        "#CCBBAA"),
]

_FEE_GROUPS = [
    ("전용 합계",  "excl_total", "#4C72B0"),
    ("EHP 합계",   "ehp_total",  "#C44E52"),
    ("공용 합계",  "comm_total", "#9EBADF"),
]


def render_electricity_view(df: pd.DataFrame) -> None:
    st.header("⚡ 전체 전기 사용내역 분석")

    buildings = sorted(df["building"].unique())
    sel_bld = st.multiselect(
        "건물 선택", ["All"] + buildings, default=["All"], key="elec_bld"
    )
    if "All" not in sel_bld and sel_bld:
        df = df[df["building"].isin(sel_bld)].copy()

    if df.empty:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
        return

    n_total = len(df)

    mc = st.columns(5)
    mc[0].metric("총 브랜드",   f"{n_total}개")
    mc[1].metric("총 부과금액", f"{df['grand_total'].sum()/1e6:.2f}M 원")
    _ep = df["grand_excl"].sum() / df["grand_total"].sum() * 100 if df["grand_total"].sum() else 0
    mc[2].metric("전용 부과",   f"{df['grand_excl'].sum()/1e6:.2f}M ({_ep:.0f}%)")
    mc[3].metric("총 사용량",   f"{int(df['kwh_total'].sum()):,} KWH")
    mc[4].metric("공용 부과",   f"{df['grand_comm'].sum()/1e6:.2f}M 원")


    # ── Anomaly flags ─────────────────────────────────────────────────────────
    _flags = pd.DataFrame(index=df["brand"].values)

    _flags["총부과 이상치"]    = df["grand_total"].values > _iqr_upper(df["grand_total"])
    _cpm2    = df["grand_total"] / df["size_m2"].replace(0, np.nan)
    _flags["면적당비용 이상치"] = _cpm2.values > _iqr_upper(_cpm2)
    _metered = df["kwh_total"] > 0
    _upm2    = (df["kwh_total"] / df["size_m2"].replace(0, np.nan)).where(_metered)
    _flags["KWH/m² 이상치"]   = _upm2.values > _iqr_upper(_upm2.dropna())

    _flag_cols = ["총부과 이상치", "면적당비용 이상치", "KWH/m² 이상치"]
    _flags.index = df["brand"].values
    _flags["플래그 수"] = _flags[_flag_cols].sum(axis=1).astype(int)
    _flags["등급"] = _flags["플래그 수"].map(
        lambda n: "🔴 위험" if n >= 2 else ("🟠 주의" if n == 1 else "🟢 정상")
    )

    tab_rank, tab_usage, tab_fee, tab_fair, tab_anom = st.tabs(
        ["순위", "사용량 구성", "요금 구성", "면적당 비용", "이상 탐지"]
    )

    # ═══════════════════════════ 순위 ═════════════════════════════════════════
    with tab_rank:
        _metric = st.radio(
            "기준", ["총부과", "전용부과", "공용부과", "사용량 (KWH)", "실효 단가 (원/KWH)"],
            horizontal=True, key="elec_rank_metric"
        )
        if _metric == "실효 단가 (원/KWH)":
            _df_eff = df[df["kwh_total"] > 0].copy()
            _df_eff["eff_rate"] = (_df_eff["grand_total"] / _df_eff["kwh_total"]).round(0)
            _col, _unit = "eff_rate", "원/KWH"
            _col_label = "실효 단가 (원/KWH)"
            _df_r = _df_eff[["brand","building","floor","size_m2","eff_rate"]].sort_values("eff_rate", ascending=True)
        else:
            _col  = {"총부과": "grand_total", "전용부과": "grand_excl",
                     "공용부과": "grand_comm", "사용량 (KWH)": "kwh_total"}[_metric]
            _unit = "원" if _metric != "사용량 (KWH)" else "KWH"
            _col_label = _metric
            _df_r = df[["brand","building","floor","size_m2",_col]].sort_values(_col, ascending=True)

        _r_up = _iqr_upper(_df_r[_col])

        fig_r = go.Figure()
        for bld in ["A", "B", "C", "D"]:
            sub = _df_r[_df_r["building"] == bld]
            if sub.empty:
                continue
            _sy = [_flag_prefix(_flags, b) + str(b)[:26] for b in sub["brand"]]
            fig_r.add_trace(go.Bar(
                x=sub[_col].values, y=_sy, name=f"{bld}동",
                orientation="h", marker_color=_BLD_COLOR[bld],
                customdata=sub[["floor", "size_m2"]].fillna("").values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    + f"{_col_label}: " + "%{x:,.0f}<br>"
                    + "층: %{customdata[0]}<br>"
                    + "면적: %{customdata[1]:.0f} m²"
                    + "<extra>%{fullData.name}</extra>"
                ),
                text=[f"{v:,.0f}" for v in sub[_col].values],
                textposition="outside" if len(_df_r) <= 25 else "none", textfont=dict(size=10),
            ))

        if _r_up < float("inf"):
            fig_r.add_vline(x=_r_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_r_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)

        fig_r.update_layout(
            height=max(420, len(_df_r) * 22 + 80),
            margin=dict(l=10, r=130, t=40, b=40),
            xaxis_title=_unit, barmode="overlay",
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            yaxis=dict(tickfont=dict(size=10), categoryorder="total ascending"),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="elec_rank_chart")

        _s = df[_col]
        sc = st.columns(4)
        sc[0].metric("합계",   f"{_s.sum():,.0f} {_unit}")
        sc[1].metric("평균",   f"{_s.mean():,.0f} {_unit}")
        sc[2].metric("중앙값", f"{_s.median():,.0f} {_unit}")
        sc[3].metric("1위",    df.loc[df[_col].idxmax(), "brand"])

    # ═══════════════════════════ 사용량 구성 ══════════════════════════════════
    with tab_usage:
        # Building summary
        st.subheader("건물별 집계")
        _bgrp = df.groupby("building").agg(
            총KWH=("kwh_total", "sum"),
            총부과=("grand_total", "sum"),
            총면적=("size_m2", "sum"),
            브랜드수=("brand", "count"),
        ).reindex(["A", "B", "C", "D"]).dropna(how="all").reset_index()

        _gc1, _gc2 = st.columns(2)
        with _gc1:
            fig_bku = go.Figure()
            for _, row in _bgrp.iterrows():
                fig_bku.add_trace(go.Bar(
                    x=[row["building"] + "동"], y=[row["총KWH"]],
                    marker_color=_BLD_COLOR.get(row["building"], "#888"),
                    text=[f"{int(row['총KWH']):,} KWH"],
                    textposition="outside", textfont=dict(size=11),
                    showlegend=False,
                ))
            fig_bku.update_layout(
                title="건물별 총 사용량 (KWH)", height=300, plot_bgcolor="white",
                yaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                margin=dict(l=10, r=10, t=50, b=30),
            )
            st.plotly_chart(fig_bku, use_container_width=True, key="elec_bld_kwh")

        with _gc2:
            fig_bkt = go.Figure()
            for _, row in _bgrp.iterrows():
                fig_bkt.add_trace(go.Bar(
                    x=[row["building"] + "동"], y=[row["총부과"]],
                    marker_color=_BLD_COLOR.get(row["building"], "#888"),
                    text=[f"{row['총부과']/1e6:.2f}M"],
                    textposition="outside", textfont=dict(size=11),
                    showlegend=False,
                ))
            fig_bkt.update_layout(
                title="건물별 총 부과금액 (원)", height=300, plot_bgcolor="white",
                yaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                margin=dict(l=10, r=10, t=50, b=30),
            )
            st.plotly_chart(fig_bkt, use_container_width=True, key="elec_bld_total")

        # Stacked KWH by source per building
        fig_bsrc = go.Figure()
        for label, col, clr in _USAGE_COLS:
            if col not in df.columns:
                continue
            _by_bld = [df[df["building"] == r["building"]][col].sum() for _, r in _bgrp.iterrows()]
            if sum(_by_bld) == 0:
                continue
            fig_bsrc.add_trace(go.Bar(
                x=[r["building"] + "동" for _, r in _bgrp.iterrows()],
                y=_by_bld, name=label, marker_color=clr,
                text=[f"{v:,.0f}" if v > 0 else "" for v in _by_bld],
                textposition="inside", textfont=dict(size=10, color="white"),
            ))
        fig_bsrc.update_layout(
            barmode="stack", title="건물별 사용량 구성 (KWH)", height=320,
            plot_bgcolor="white",
            yaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            legend=dict(orientation="h", y=1.12),
            margin=dict(l=10, r=10, t=70, b=30),
        )
        st.plotly_chart(fig_bsrc, use_container_width=True, key="elec_bld_src")

        _bgrp_disp = _bgrp.copy()
        _bgrp_disp["총KWH"]  = _bgrp_disp["총KWH"].apply(lambda v: f"{int(v):,} KWH")
        _bgrp_disp["총부과"] = _bgrp_disp["총부과"].apply(lambda v: f"{v/1e6:.2f}M 원")
        _bgrp_disp["KWH/m²"] = (_bgrp["총KWH"] / _bgrp["총면적"]).apply(lambda v: f"{v:.1f}")
        _bgrp_disp["원/m²"]  = (_bgrp["총부과"] / _bgrp["총면적"]).apply(lambda v: f"{v:,.0f}")
        _bgrp_disp["building"] = _bgrp_disp["building"] + "동"
        _bgrp_disp = _bgrp_disp.rename(columns={
            "building": "건물", "브랜드수": "브랜드",
            "총KWH": "사용량 합계", "총부과": "부과 합계",
        })
        st.dataframe(_bgrp_disp[["건물","브랜드","사용량 합계","부과 합계","KWH/m²","원/m²"]],
                     use_container_width=True, hide_index=True)

        st.divider()

        # Brand-level usage stacked bar
        _n_show = st.slider("상위 N개 브랜드", 10, min(60, n_total), min(30, n_total),
                            key="elec_usage_n")
        _top = df.nlargest(_n_show, "kwh_total").sort_values("kwh_total", ascending=True)
        _ty  = [_flag_prefix(_flags, b) + str(b)[:26] for b in _top["brand"]]

        fig_us = go.Figure()
        for label, col, clr in _USAGE_COLS:
            if col not in df.columns:
                continue
            _vals = _top[col].fillna(0).values
            if _vals.sum() == 0:
                continue
            fig_us.add_trace(go.Bar(
                x=_vals, y=_ty, name=label, orientation="h",
                marker_color=clr,
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:,.0f} KWH<extra></extra>",
                text=[f"{v:,.0f}" if v >= 100 else "" for v in _vals],
                textposition="inside", textfont=dict(size=9, color="white"),
            ))
        fig_us.update_layout(
            barmode="stack",
            title=f"상위 {_n_show}개 브랜드 사용량 구성 (KWH)",
            height=max(420, _n_show * 22 + 80),
            margin=dict(l=10, r=20, t=60, b=40),
            xaxis_title="KWH", plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            legend=dict(orientation="h", y=1.06),
        )
        st.plotly_chart(fig_us, use_container_width=True, key="elec_usage_stacked")

        # EHP vs 비-EHP comparison
        st.divider()
        st.subheader("EHP 유무 비교")
        df_ehp  = df[df["kwh_ehp"] > 0].copy()
        df_noehp = df[df["kwh_ehp"] == 0].copy()
        st.caption(f"EHP 있음: **{len(df_ehp)}개** 브랜드  |  EHP 없음: **{len(df_noehp)}개** 브랜드")

        _cmp_metrics = {
            "총부과 (원)":      ("grand_total", 1),
            "전용부과 (원)":    ("grand_excl",  1),
            "EHP요금 (원)":     ("ehp_total",   1),
            "총 KWH":           ("kwh_total",   1),
            "EHP KWH":          ("kwh_ehp",     1),
        }
        _cmp_rows = []
        for label, (col, _) in _cmp_metrics.items():
            if col not in df.columns: continue
            _cmp_rows.append({
                "항목":        label,
                "EHP 있음 평균":  f"{df_ehp[col].mean():,.0f}",
                "EHP 없음 평균":  f"{df_noehp[col].mean():,.0f}" if len(df_noehp) > 0 else "-",
                "EHP 있음 합계":  f"{df_ehp[col].sum():,.0f}",
            })
        st.dataframe(pd.DataFrame(_cmp_rows), use_container_width=True, hide_index=True)

        # Side-by-side box approximation using bar (mean + range)
        _comp_bar_cols = [("총부과","grand_total","#4C72B0"),("총 KWH","kwh_total","#DD8A00")]
        _bc1, _bc2 = st.columns(2)
        for (label, col, clr), _bcol in zip(_comp_bar_cols, [_bc1, _bc2]):
            with _bcol:
                _grp = pd.DataFrame({
                    "그룹": ["EHP 있음", "EHP 없음"],
                    "평균": [df_ehp[col].mean(), df_noehp[col].mean() if len(df_noehp)>0 else 0],
                    "중앙값": [df_ehp[col].median(), df_noehp[col].median() if len(df_noehp)>0 else 0],
                })
                fig_cmp = go.Figure()
                fig_cmp.add_trace(go.Bar(x=_grp["그룹"], y=_grp["평균"], name="평균",
                                         marker_color=[clr,"#AAAAAA"],
                                         text=[f"{v:,.0f}" for v in _grp["평균"]],
                                         textposition="outside"))
                fig_cmp.update_layout(title=f"{label} 평균 비교", height=280, plot_bgcolor="white",
                                      yaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                                      showlegend=False, margin=dict(l=10,r=10,t=50,b=30))
                st.plotly_chart(fig_cmp, use_container_width=True, key=f"elec_ehp_cmp_{col}")

    # ═══════════════════════════ 요금 구성 ════════════════════════════════════
    with tab_fee:
        _fview = st.radio("보기", ["전체 구성 donut", "브랜드별 stacked"],
                          horizontal=True, key="elec_fee_view")

        if _fview == "전체 구성 donut":
            _gc1, _gc2 = st.columns(2)
            with _gc1:
                # Top-level: excl / ehp / comm
                _d1 = {
                    "전용 합계":  df["excl_total"].sum(),
                    "EHP 합계":   df["ehp_total"].sum(),
                    "공용 합계":  df["comm_total"].sum(),
                }
                fig_d1 = go.Figure(go.Pie(
                    labels=list(_d1.keys()), values=list(_d1.values()), hole=0.45,
                    marker=dict(colors=["#4C72B0", "#C44E52", "#9EBADF"]),
                    textinfo="label+percent", textfont=dict(size=12),
                ))
                fig_d1.update_layout(title="전용 / EHP / 공용", height=360,
                                     margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_d1, use_container_width=True, key="elec_donut1")

            with _gc2:
                # Fee component breakdown (base / energy / climate / fund)
                _d2 = {
                    "기본요금":     df["excl_base"].sum() + df["ehp_base"].sum() + df["comm_base"].sum(),
                    "사용요금":     df["excl_energy"].sum() + df["ehp_energy"].sum() + df["comm_energy"].sum(),
                    "기후변화요금": df["excl_climate"].sum() + df["ehp_climate"].sum() + df["comm_climate"].sum(),
                    "전력기금":     df["excl_fund"].sum() + df["ehp_fund"].sum() + df["comm_fund"].sum(),
                }
                fig_d2 = go.Figure(go.Pie(
                    labels=list(_d2.keys()), values=list(_d2.values()), hole=0.45,
                    marker=dict(colors=["#4C72B0", "#C44E52", "#DD8A00", "#8172B2"]),
                    textinfo="label+percent", textfont=dict(size=12),
                ))
                fig_d2.update_layout(title="요금 항목별 구성", height=360,
                                     margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_d2, use_container_width=True, key="elec_donut2")

            _tot_fee = df["grand_total"].sum()
            _tbl = pd.DataFrame({
                "항목": ["전용 합계", "EHP 합계", "공용 합계"],
                "금액 (원)": [f"{df[c].sum():,.0f}" for c in ["excl_total","ehp_total","comm_total"]],
                "비중": [f"{df[c].sum()/_tot_fee*100:.1f}%" for c in ["excl_total","ehp_total","comm_total"]],
            })
            st.dataframe(_tbl, use_container_width=True, hide_index=True)

            # 기후변화요금 비중 highlight
            st.divider()
            _climate_total = df["excl_climate"].sum() + df["ehp_climate"].sum() + df["comm_climate"].sum()
            _climate_pct   = _climate_total / _tot_fee * 100 if _tot_fee else 0
            st.info(f"**기후변화요금** 총 {_climate_total/1e6:.2f}M 원 — 전체 전기요금의 **{_climate_pct:.1f}%**")

            # 역률요금 분석
            st.subheader("역률요금 (역률 조정 할인/할증)")
            st.caption("역률요금이 음수인 브랜드는 역률 개선으로 할인을 받고, 양수는 저역률 패널티입니다.")
            _pf_cols = [c for c in ["excl_pfactor","ehp_pfactor","comm_pfactor"] if c in df.columns]
            if _pf_cols:
                df_pf = df[["brand","building"] + _pf_cols].copy()
                df_pf["역률요금_합계"] = df_pf[_pf_cols].sum(axis=1)
                df_pf = df_pf[df_pf["역률요금_합계"] != 0].sort_values("역률요금_합계")
                if not df_pf.empty:
                    _pf_colors = ["#4C72B0" if v < 0 else "#C44E52" for v in df_pf["역률요금_합계"].values]
                    fig_pf = go.Figure(go.Bar(
                        x=df_pf["역률요금_합계"].values,
                        y=[str(b)[:26] for b in df_pf["brand"]],
                        orientation="h", marker_color=_pf_colors,
                        text=[f"{v:,.0f}" for v in df_pf["역률요금_합계"].values],
                        textposition="outside", textfont=dict(size=10),
                    ))
                    fig_pf.add_vline(x=0, line_color="#888", line_width=1)
                    fig_pf.update_layout(
                        height=max(320, len(df_pf)*22+80), xaxis_title="원 (음수=할인, 양수=할증)",
                        plot_bgcolor="white", xaxis=dict(gridcolor="#DDDDDD",griddash="dot"),
                        margin=dict(l=10,r=130,t=40,b=40),
                    )
                    st.plotly_chart(fig_pf, use_container_width=True, key="elec_pfactor_chart")
                    _disc = df_pf[df_pf["역률요금_합계"] < 0]
                    _surch = df_pf[df_pf["역률요금_합계"] > 0]
                    pc = st.columns(3)
                    pc[0].metric("할인 브랜드", f"{len(_disc)}개", delta=f"{_disc['역률요금_합계'].sum():,.0f} 원")
                    pc[1].metric("할증 브랜드", f"{len(_surch)}개", delta=f"{_surch['역률요금_합계'].sum():,.0f} 원", delta_color="inverse")
                    pc[2].metric("순 역률요금", f"{df_pf['역률요금_합계'].sum():,.0f} 원")

        else:
            _n_show2 = st.slider("상위 N개 브랜드", 10, min(60, n_total), min(30, n_total),
                                 key="elec_fee_n")
            _top2 = df.nlargest(_n_show2, "grand_total").sort_values("grand_total", ascending=True)
            _fy   = [_flag_prefix(_flags, b) + str(b)[:26] for b in _top2["brand"]]

            fig_fs = go.Figure()
            for label, col, clr in _FEE_GROUPS:
                if col not in df.columns:
                    continue
                fig_fs.add_trace(go.Bar(
                    x=_top2[col].fillna(0).values, y=_fy, name=label,
                    orientation="h", marker_color=clr,
                    text=[f"{v/1e6:.2f}M" if v >= 1e5 else "" for v in _top2[col].fillna(0).values],
                    textposition="inside", textfont=dict(size=9, color="white"),
                ))
            fig_fs.update_layout(
                barmode="stack",
                height=max(420, _n_show2 * 22 + 80),
                margin=dict(l=10, r=20, t=40, b=40),
                xaxis_title="원", plot_bgcolor="white",
                xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
                legend=dict(orientation="h", y=1.02),
            )
            st.plotly_chart(fig_fs, use_container_width=True, key="elec_fee_stacked")

    # ═══════════════════════════ 면적당 비용 ══════════════════════════════════
    with tab_fair:
        _u = st.radio("면적 단위", ["㎡", "평"], horizontal=True, key="elec_fair_unit")
        _a_col = "size_m2" if _u == "㎡" else "size_py"

        _df_f = df[["brand", "building", "grand_total", _a_col]].copy()
        _df_f = _df_f[_df_f[_a_col] > 0].copy()
        _df_f["cpa"] = _df_f["grand_total"] / _df_f[_a_col]
        _df_f = _df_f.sort_values("cpa", ascending=True)

        _sf = _df_f["cpa"]
        _f_up = _iqr_upper(_sf)
        _q1f, _q3f = _sf.quantile(0.25), _sf.quantile(0.75)
        _f_lo = max(0.0, float(_q1f - 1.5 * (_q3f - _q1f)))

        _bord_clr = [
            "#8B1A1A" if v > _f_up else ("#1A5C2A" if _f_lo > 0 and v < _f_lo else "white")
            for v in _sf.values
        ]
        _bord_w = [2.5 if v > _f_up or (_f_lo > 0 and v < _f_lo) else 0 for v in _sf.values]

        fig_f = go.Figure(go.Bar(
            x=_sf.values,
            y=[_flag_prefix(_flags, b) + str(b)[:26] for b in _df_f["brand"]],
            orientation="h",
            marker_color=[_BLD_COLOR.get(b, "#888") for b in _df_f["building"]],
            marker_line=dict(color=_bord_clr, width=_bord_w),
            text=[f"{v:,.0f}" for v in _sf.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig_f.add_vline(x=float(_sf.median()), line_dash="dash", line_color="#C44E52",
                        annotation_text=f"중앙값 {_sf.median():,.0f}",
                        annotation_position="top right", annotation_font_size=10)
        if _f_up < _sf.max() * 5:
            fig_f.add_vline(x=_f_up, line_dash="dot", line_color="#DD8A00",
                            annotation_text=f"IQR 상한 {_f_up:,.0f}",
                            annotation_position="top left", annotation_font_size=10)
        fig_f.update_layout(
            height=max(420, len(_df_f) * 22 + 80),
            xaxis_title=f"원/{_u}", plot_bgcolor="white",
            xaxis=dict(gridcolor="#DDDDDD", griddash="dot"),
            margin=dict(l=10, r=150, t=40, b=40),
        )
        st.plotly_chart(fig_f, use_container_width=True, key="elec_fair_chart")

        fc = st.columns(4)
        fc[0].metric("중앙값",    f"{_sf.median():,.0f} 원/{_u}")
        fc[1].metric("평균",      f"{_sf.mean():,.0f} 원/{_u}")
        fc[2].metric("IQR 상한",  f"{_f_up:,.0f} 원/{_u}")
        fc[3].metric("상한 초과", f"{int((_sf > _f_up).sum())}개")

    # ═══════════════════════════ 이상 탐지 ════════════════════════════════════
    with tab_anom:
        n_crit  = int((_flags["플래그 수"] >= 2).sum())
        n_watch = int((_flags["플래그 수"] == 1).sum())
        n_ok    = int((_flags["플래그 수"] == 0).sum())

        ac = st.columns(3)
        ac[0].metric("🔴 위험 (≥2 플래그)", f"{n_crit}개")
        ac[1].metric("🟠 주의 (1 플래그)",   f"{n_watch}개")
        ac[2].metric("🟢 정상",              f"{n_ok}개")
        st.divider()

        _hm = _flags.sort_values("플래그 수", ascending=False).head(40)
        _hm_brands = list(_hm.index)
        _hm_z    = [[int(_hm.at[b, c]) for c in _flag_cols] for b in _hm_brands]
        _hm_text = [["✓" if _hm.at[b, c] else "" for c in _flag_cols] for b in _hm_brands]

        fig_hm = go.Figure(go.Heatmap(
            z=_hm_z, x=_flag_cols, y=_hm_brands,
            colorscale=[[0, "#F0F0F0"], [1, "#C44E52"]],
            showscale=False,
            text=_hm_text, texttemplate="%{text}",
            textfont=dict(size=13, color="white"),
            xgap=3, ygap=3,
        ))
        fig_hm.update_layout(
            height=max(300, len(_hm_brands) * 22 + 80),
            margin=dict(l=10, r=20, t=40, b=40),
            plot_bgcolor="white",
            yaxis=dict(autorange="reversed", tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_hm, use_container_width=True, key="elec_anom_heatmap")

        _flagged_idx = _flags[_flags["플래그 수"] > 0].index
        if len(_flagged_idx) > 0:
            st.subheader("플래그 상세")
            _detail = df[df["brand"].isin(_flagged_idx)][
                ["brand", "building", "floor", "size_m2", "kwh_total",
                 "grand_excl", "grand_comm", "grand_total"]
            ].copy()
            _meta = _flags[["플래그 수", "등급"]].reset_index().rename(columns={"index": "brand"})
            _detail = _detail.merge(_meta, on="brand", how="left")
            _detail = _detail.sort_values("플래그 수", ascending=False).reset_index(drop=True)
            st.dataframe(
                _detail,
                column_config={
                    "brand":       st.column_config.TextColumn("브랜드"),
                    "building":    st.column_config.TextColumn("건물"),
                    "floor":       st.column_config.TextColumn("층"),
                    "size_m2":     st.column_config.NumberColumn("면적 (m²)", format="%.0f"),
                    "kwh_total":   st.column_config.ProgressColumn(
                        "KWH", format="%,.0f", min_value=0,
                        max_value=int(_detail["kwh_total"].max()) if not _detail["kwh_total"].empty and _detail["kwh_total"].max() > 0 else 1,
                    ),
                    "grand_excl":  st.column_config.NumberColumn("전용부과 (원)", format="%,.0f"),
                    "grand_comm":  st.column_config.NumberColumn("공용부과 (원)", format="%,.0f"),
                    "grand_total": st.column_config.NumberColumn("총부과 (원)", format="%,.0f"),
                    "플래그 수":   st.column_config.NumberColumn("플래그", format="%d"),
                    "등급":        st.column_config.TextColumn("등급"),
                },
                use_container_width=True,
                hide_index=True,
            )
