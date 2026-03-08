"""
report.py  —  Business-ready PDF report generator (English / Korean)
"""
import io
import os
import textwrap
from datetime import date

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D as _Line2D
from matplotlib.font_manager import FontProperties as _FontProperties
from matplotlib import font_manager as _fm
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas as _CanvasBase
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether,
)

from data import to_numeric_series

# ── Korean font setup ────────────────────────────────────────────────────────

_FONT_DIR  = os.path.expanduser("~/.fonts")
_FONT_REG  = os.path.join(_FONT_DIR, "NanumGothic-Regular.ttf")
_FONT_BOLD = os.path.join(_FONT_DIR, "NanumGothic-Bold.ttf")
_FONT_URLS = {
    _FONT_REG:  "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf",
    _FONT_BOLD: "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Bold.ttf",
}

def _ensure_fonts():
    import urllib.request
    os.makedirs(_FONT_DIR, exist_ok=True)
    for path, url in _FONT_URLS.items():
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
    pdfmetrics.registerFont(TTFont("NanumGothic",      _FONT_REG))
    pdfmetrics.registerFont(TTFont("NanumGothic-Bold", _FONT_BOLD))
    from reportlab.pdfbase.pdfmetrics import registerFontFamily
    registerFontFamily("NanumGothic",
                       normal="NanumGothic", bold="NanumGothic-Bold",
                       italic="NanumGothic", boldItalic="NanumGothic-Bold")
    # Delete stale matplotlib font cache so new fonts are picked up
    import glob as _glob
    import matplotlib as _mpl
    for _fc in _glob.glob(os.path.join(_mpl.get_cachedir(), "fontlist-*.json")):
        try:
            os.remove(_fc)
        except OSError:
            pass

    _fm.fontManager.addfont(_FONT_REG)
    _fm.fontManager.addfont(_FONT_BOLD)

    # Clear the findfont LRU cache so lookups use the updated font list
    try:
        _fm.findfont.cache_clear()
    except AttributeError:
        pass

    _font_name = _fm.FontProperties(fname=_FONT_REG).get_name()
    plt.rcParams["font.sans-serif"] = [_font_name] + [
        f for f in plt.rcParams.get("font.sans-serif", []) if f != _font_name
    ]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

# ── Palette ──────────────────────────────────────────────────────────────────

C_NAVY     = colors.HexColor("#1B2A3B")
C_BLUE     = colors.HexColor("#2E6DA4")
C_LIGHT    = colors.HexColor("#EEF3FA")
C_DIVIDER  = colors.HexColor("#D0D8E4")
C_TEXT     = colors.HexColor("#2D2D2D")
C_SUBTEXT  = colors.HexColor("#666666")
C_WHITE    = colors.white

C_CRITICAL = colors.HexColor("#E63946")
C_WATCH    = colors.HexColor("#F4882A")
C_ALERT    = colors.HexColor("#E8B84B")
C_STABLE   = colors.HexColor("#43AA6F")
C_NORMAL   = colors.HexColor("#888888")
C_NODATA   = colors.HexColor("#94B8D0")

M_CRITICAL = "#E63946"
M_WATCH    = "#F4882A"
M_ALERT    = "#E8B84B"
M_STABLE   = "#43AA6F"
M_NORMAL   = "#888888"
M_NODATA   = "#94B8D0"
M_BAR      = "#4C72B0"

STATUS_COLOR_RL = {
    "Critical": C_CRITICAL, "Watch": C_WATCH,
    "Alert":    C_ALERT,    "Stable": C_STABLE,
    "Normal":   C_NORMAL,   "No Data": C_NODATA,
}
STATUS_COLOR_M = {
    "Critical": M_CRITICAL, "Watch": M_WATCH,
    "Alert":    M_ALERT,    "Stable": M_STABLE,
    "Normal":   M_NORMAL,   "No Data": M_NODATA,
}
STATUS_ORDER = {
    "Critical": 0, "Watch": 1, "Alert": 2,
    "Normal": 3, "Stable": 4, "No Data": 5,
}

UTILITY_META = {
    "water":  {"en": "Water",       "ko": "수도",  "unit": "m³"},
    "hwater": {"en": "Hot Water",   "ko": "온수",  "unit": "m³"},
    "elect":  {"en": "Electricity", "ko": "전기",  "unit": "kWh"},
    "heat":   {"en": "Heat",        "ko": "열",    "unit": "m³/MWh"},
}

# ── Translations ─────────────────────────────────────────────────────────────

_STRINGS = {
    "en": {
        "report_title":    "Utility Usage Report",
        "report_subtitle": "Tenant Analysis & Consumption Summary",
        "date":            "Date",
        "buildings":       "Buildings",
        "floors":          "Floors",
        "tenants_n":       "Tenants Analyzed",
        "threshold_label": "Alert Threshold",
        "threshold_value": "Bottom / Top {pct}% of distribution",
        "utilities":       "Utilities Covered",
        "summary_title":   "Summary by Utility",
        # overview table
        "col_utility":  "Utility",
        "col_tenants":  "Tenants w/ Data",
        "col_avg_use":  "Avg Usage",
        "col_avg_pct":  "Avg Change (%)",
        "col_critical": "Critical",
        "col_watch":    "Watch",
        "col_alert":    "Alert",
        "col_status":   "Status",
        # status display labels (for charts & tables)
        "Critical": "Critical",
        "Watch":    "Watch",
        "Alert":    "Alert",
        "Stable":   "Stable",
        "Normal":   "Normal",
        "No Data":  "No Data",
        # overview row status
        "ov_critical": "Needs Attention",
        "ov_watch":    "Monitor",
        "ov_normal":   "Normal",
        # cover note
        "cover_note": (
            "<b>Critical</b> = high usage AND large increase &nbsp;&nbsp;"
            "<b>Watch</b> = high usage, moderate % change &nbsp;&nbsp;"
            "<b>Alert</b> = sharp % rise from smaller base<br/>"
            "Thresholds are set at the bottom/top {pct}% of each utility's distribution."
        ),
        # section headers
        "section_analysis": "  {name}  —  Usage Analysis",
        "section_detail":   "  {name}  —  Tenant Detail",
        # distribution
        "dist_title": "Distribution Summary",
        "dist_note":  "This table describes how usage is spread across all tenants for this billing period.",
        "col_metric":  "Metric",
        "col_value":   "Value",
        "col_meaning": "What This Means",
        # stat rows
        "stat_n":       "Tenants with data",
        "stat_n_m":     "Number of tenants where both this period and the previous period have readings.",
        "stat_med":     "Median usage",
        "stat_med_m":   "Half of all tenants used less than this, half used more. More reliable than the average when a few tenants have very high usage.",
        "stat_avg":     "Average usage",
        "stat_avg_m":   "Simple average across all tenants. Can be pulled upward by a few heavy users.",
        "stat_std":     "Spread (std. deviation)",
        "stat_std_m":   "How different tenants are from each other. A larger number means a wider gap between the lowest and highest users.",
        "stat_p80":     "Top 20% threshold",
        "stat_p80_m":   "Tenants using more than this are in the highest-usage 20%. Worth checking whether this matches their business type.",
        "stat_p20":     "Bottom 20% threshold",
        "stat_p20_m":   "Tenants using less than this are in the lowest-usage 20%.",
        "stat_hi":      "High-change threshold (top {pct}%)",
        "stat_hi_m":    "Tenants whose usage increased by more than this amount are in the top {pct}% of movers — flagged for review.",
        "stat_lo":      "Low-change threshold (bottom {pct}%)",
        "stat_lo_m":    "Tenants whose usage decreased by more than this amount are in the bottom {pct}% of movers.",
        # charts
        "charts_title":       "Usage Distribution & Change Overview",
        "chart_hist_title":   "{name} — Current Usage Distribution",
        "chart_bar_title":    "{name} — Usage Change per Tenant",
        "chart_donut_title":  "Tenant Status Breakdown",
        "chart_hist_caption": (
            "Figure: Distribution of current {name} usage. "
            "Amber bars = bottom/top {pct}%. Red dashed line = median."
        ),
        "chart_bar_caption": (
            "Figure: Each bar shows a tenant's usage change vs. last period, colored by status. "
            "Donut shows proportion of tenant statuses."
        ),
        # detail
        "flagged_title": "Flagged Tenants",
        "flagged_note":  "These tenants require follow-up. They fall in the top or bottom {pct}% of usage change or percentage change for this utility.",
        "legend_title":  "Status Guide",
        "all_title":     "All Tenants",
        # detail table headers
        "th_no":     "#",
        "th_tenant": "Tenant",
        "th_bldg":   "Bldg",
        "th_floor":  "Floor",
        "th_last":   "Last ({unit})",
        "th_this":   "This ({unit})",
        "th_change": "Change ({unit})",
        "th_pct":    "Chg (%)",
        "th_status": "Status",
        # legend headers
        "th_leg_status":    "Status",
        "th_leg_meaning":   "What It Means",
        "th_leg_condition": "Definition",
        "th_leg_action":    "Recommended Action",
        # status descriptions & actions
        "desc_Critical": "High usage AND large increase",
        "desc_Watch":    "High overall usage, moderate % change",
        "desc_Alert":    "Sharp % rise from a smaller base",
        "desc_Stable":   "Low usage and declining",
        "desc_Normal":   "Within expected range",
        "desc_No Data":  "No previous period data",
        # plain-language conditions ({pct} is substituted at render time)
        "cond_Critical": "Absolute change AND % change both exceed the top {pct}% threshold",
        "cond_Watch":    "Absolute change OR % change exceeds the top {pct}% threshold — but not both simultaneously",
        "cond_Alert":    "Absolute change OR % change falls below the bottom {pct}% threshold — but not both simultaneously",
        "cond_Stable":   "Absolute change AND % change both fall below the bottom {pct}% threshold",
        "cond_Normal":   "Within the middle range — no threshold exceeded in either direction",
        "cond_No Data":  "No prior-period reading; % change cannot be computed",
        "leg_condition_note": (
            "Thresholds are computed independently per utility using only tenants "
            "with readings in both periods. "
            "Top {pct}% threshold = {pct}th percentile of absolute change (high side). "
            "Bottom {pct}% threshold = {pct}th percentile of absolute change (low side). "
            "The same percentile logic applies to % change."
        ),
        "act_Critical":  "Investigate immediately — check for leaks, equipment faults, or unusual activity",
        "act_Watch":     "Monitor — usage is elevated; confirm it matches the business type",
        "act_Alert":     "Verify cause — new equipment, new staff, or possible reporting error",
        "act_Stable":    "No action needed",
        "act_Normal":    "No action needed",
        "act_No Data":   "Confirm new tenant or first full billing period",
        # footer / back
        "footer_left": "Utility Analysis Report  ·  Confidential",
        "footer_page": "Page {n} of {total}",
        "end_title":   "End of Report",
        "end_note":    (
            "Generated on {date}. This report is intended for internal management use only. "
            "All consumption figures are based on meter readings provided in the uploaded data file."
        ),
        # executive summary
        "exec_title": "Executive Summary",
        "exec_all_clear": (
            "All <b>{n_total}</b> tenants across {n_util} utilities are within expected usage ranges this period. "
            "No immediate action is required."
        ),
        "exec_critical_lead": (
            "<b>{n_critical}</b> tenant(s) require <b>immediate attention</b> this period."
        ),
        "exec_building_concentration": (
            " Building <b>{bldg}</b> accounts for the highest number of flags ({n} tenant(s))."
        ),
        "exec_vacancy": (
            " {n} unit(s) show zero usage despite prior-period activity — possible vacancy."
        ),
        "exec_new": (
            " {n} new tenant(s) are in their first billing period and will appear in future comparisons."
        ),
        "exec_watch_alert": (
            " Additionally, {n_watch} tenant(s) are flagged Watch and {n_alert} tenant(s) are flagged Alert "
            "across all utilities."
        ),
        # action checklist
        "checklist_title":  "Action Checklist",
        "checklist_note":   (
            "Use this page to track follow-up actions for all flagged tenants. "
            "Mark each item once investigated and resolved."
        ),
        "cl_th_no":      "#",
        "cl_th_tenant":  "Tenant",
        "cl_th_bldg":    "Bldg / Floor",
        "cl_th_utility": "Utility",
        "cl_th_status":  "Status",
        "cl_th_issue":   "Issue",
        "cl_th_check":   "Checked",
        "cl_th_resolve": "Resolved",
        "cl_th_remarks": "Remarks",
        "cl_none":       "No flagged tenants this period. No action required.",
        "cl_issue_critical": "Usage +{change} ({pct}) — high absolute & high % change",
        "cl_issue_alert":    "Usage +{change} ({pct}) — sharp % rise from smaller base",
        "cl_issue_watch":    "Usage +{change} ({pct}) — elevated absolute usage",
        # building totals
        "bldg_totals_title": "Usage by Building",
        "col_bldg":   "Building",
        "col_curr_t": "This Period",
        "col_prev_t": "Last Period",
        "col_chg_t":  "Change",
        "col_pct_t":  "Chg (%)",
        # data coverage
        "coverage_title": "Data Coverage",
        "col_utility_c":  "Utility",
        "col_total_c":    "Total Tenants",
        "col_with_data":  "With Data",
        "col_no_data_c":  "No Prior Data",
        "col_coverage":   "Coverage",
        # top 10 consumers
        "top10_title": "Top 10 Highest Current Usage",
        "top10_note":  "Tenants with the highest absolute consumption this period, regardless of change status.",
        "th_rank":     "#",
        "th_curr_use": "Current Usage",
        # critical profile cards
        "profile_title": "Critical Tenant Profiles",
        "profile_note":  (
            "Cross-utility summary for each Critical tenant. "
            "Each row shows their status and change for every utility in this report."
        ),
        "profile_none":  "No Critical tenants this period.",
        # seasonal note
        "seasonal_title":  "Seasonal Note",
        "seasonal_spring": (
            "This report covers a spring transition period (Mar–May). "
            "Increases in cooling and decreases in heating are expected — "
            "weight heating flags accordingly."
        ),
        "seasonal_summer": (
            "This report covers a summer period (Jun–Aug). "
            "Elevated electricity and cooling usage is expected due to air conditioning load."
        ),
        "seasonal_fall": (
            "This report covers an autumn transition period (Sep–Nov). "
            "Heating usage typically rises this period — verify that heat spikes "
            "are not simply seasonal before escalating."
        ),
        "seasonal_winter": (
            "This report covers a winter period (Dec–Feb). "
            "Heating consumption is expected to be at its seasonal peak. "
            "Compare against the same period last year for a more meaningful benchmark."
        ),
        # building comparison chart
        "bldg_chart_title": "{name} — Average Usage by Building",
        "bldg_chart_caption": "Figure: Average current-period usage per building. Darker bar = higher than building-wide average.",
        # critical alerts page
        "critical_title": "Critical Alerts",
        "critical_intro": (
            "The tenants listed below require <b>immediate attention</b>. "
            "They fall in the top {pct}% for both absolute usage change and percentage change — "
            "indicating an unusual spike in consumption this period that warrants investigation."
        ),
        "critical_none": (
            "No critical alerts this period. "
            "All tenants are within expected usage ranges."
        ),
        "critical_section": "{name}  —  {n} Critical Tenant(s)",
        "critical_interp": (
            "Out of <b>{total}</b> tenants, <b>{n}</b> are flagged Critical for {utility}. "
            "Their average usage this period is <b>{avg_crit}</b>, "
            "which is <b>{comparison}</b> the building-wide average of {avg_all}. "
            "The largest single increase was recorded by <b>{top_tenant}</b>: "
            "<b>{top_change} {unit}</b> ({top_pct} change). "
            "Immediate investigation is recommended."
        ),
        "critical_watch_note": (
            "Additionally, {n_watch} tenant(s) are flagged <b>Watch</b> and {n_alert} tenant(s) are flagged <b>Alert</b> "
            "for this utility. See the detailed section for the full list."
        ),
        # special analysis
        "section_special":   "  Special Analysis  —  Vacancy & Unusual Activity",
        "special_intro": (
            "This section highlights tenants that may warrant additional investigation beyond the standard "
            "Critical / Alert flags — including possible vacancies, new tenants in their first billing period, "
            "tenants flagged across multiple utilities simultaneously, and tenants showing a sudden sharp drop in usage."
        ),
        "sub_vacancy":    "Suspected Vacant Units",
        "vacancy_note":   "Tenants whose current-period usage is zero or near-zero for at least one utility, "
                          "despite having prior-period data. This may indicate a unit has become vacant or a business has closed.",
        "vacancy_none":   "No tenants with zero current usage detected.",
        "sub_new_tenants":"New Tenants / First Billing Period",
        "new_tenants_note": "Tenants with no prior-period data (first occupancy or first full billing cycle). "
                            "Flagged as 'No Data' — no change analysis is possible yet.",
        "new_none":         "No new tenants detected.",
        "sub_missing_data": "Missing Meter Data",
        "missing_data_note":"Tenants with no current-period reading recorded for at least one utility. "
                            "Verify whether the meter was skipped during collection or the unit is inactive.",
        "missing_none":     "No missing data detected.",
        "sub_multi_flag": "Multi-Utility Alerts",
        "multi_flag_note":"Tenants flagged Critical, Watch, or Alert in two or more utility categories simultaneously. "
                          "A simultaneous spike across water, electricity, and heating may indicate a systematic issue.",
        "multi_flag_none":"No tenants are flagged across multiple utilities simultaneously.",
        "sub_sharp_drop": "Tenants with Sudden Usage Drop",
        "sharp_drop_note":"Tenants classified as Stable (low usage AND sharp decline) in two or more utilities. "
                          "A sudden drop across multiple utilities can signal vacancy, business closure, or a meter fault.",
        "sharp_drop_none":"No tenants show a sudden drop across multiple utilities.",
        "th_utilities":   "Utilities",
        # median vs mean interpretation
        "med_vs_avg_high": (
            "The average ({avg}) is notably higher than the median ({med}). "
            "This means a small number of high-usage tenants are pulling the average up — "
            "use the <b>median</b> as the more representative baseline for a typical tenant in this building."
        ),
        "med_vs_avg_low": (
            "The average ({avg}) is notably lower than the median ({med}). "
            "Some tenants with near-zero usage are pulling the average down — "
            "check whether those units are currently occupied."
        ),
        "med_vs_avg_even": (
            "The average ({avg}) and median ({med}) are close, "
            "indicating a fairly even usage distribution with no extreme outliers skewing the data."
        ),
    },
    "ko": {
        "report_title":    "관리비 사용량 분석 보고서",
        "report_subtitle": "임차인별 소비량 분석 요약",
        "date":            "보고서 날짜",
        "buildings":       "건물",
        "floors":          "층",
        "tenants_n":       "분석 대상 임차인 수",
        "threshold_label": "이상치 기준",
        "threshold_value": "하위 / 상위 {pct}% 기준",
        "utilities":       "분석 항목",
        "summary_title":   "항목별 요약",
        "col_utility":  "항목",
        "col_tenants":  "비교 가능 임차인",
        "col_avg_use":  "평균 사용량",
        "col_avg_pct":  "평균 변화율 (%)",
        "col_critical": "과다",
        "col_watch":    "주의",
        "col_alert":    "경보",
        "col_status":   "상태",
        "Critical": "과다",
        "Watch":    "주의",
        "Alert":    "경보",
        "Stable":   "안정",
        "Normal":   "정상",
        "No Data":  "데이터 없음",
        "ov_critical": "즉시 확인 필요",
        "ov_watch":    "모니터링 필요",
        "ov_normal":   "정상",
        "cover_note": (
            "<b>과다</b> = 사용량 급증 및 절대량 높음 &nbsp;&nbsp;"
            "<b>주의</b> = 전체 사용량 높음, 변화율 보통 &nbsp;&nbsp;"
            "<b>경보</b> = 기준 대비 급격한 변화율<br/>"
            "기준은 각 항목 분포의 하위/상위 {pct}%입니다."
        ),
        "section_analysis": "  {name}  —  사용량 분석",
        "section_detail":   "  {name}  —  임차인 상세",
        "dist_title": "사용량 분포 요약",
        "dist_note":  "이번 청구 기간 전체 임차인의 사용량 분포를 나타냅니다.",
        "col_metric":  "지표",
        "col_value":   "값",
        "col_meaning": "해석",
        "stat_n":     "비교 가능 임차인 수",
        "stat_n_m":   "이번 기간과 이전 기간 모두 검침 데이터가 있는 임차인 수입니다.",
        "stat_med":   "중앙값 사용량",
        "stat_med_m": "전체 임차인의 절반은 이 값보다 적게, 나머지 절반은 더 많이 사용했습니다. 일부 고사용 임차인의 영향을 덜 받아 평균보다 신뢰도가 높습니다.",
        "stat_avg":   "평균 사용량",
        "stat_avg_m": "전체 임차인의 단순 평균입니다. 고사용 임차인이 있을 경우 실제보다 높게 나타날 수 있습니다.",
        "stat_std":   "편차 (표준편차)",
        "stat_std_m": "임차인 간 사용량 차이를 나타냅니다. 값이 클수록 임차인 간 사용량 편차가 큰 것입니다.",
        "stat_p80":   "상위 20% 기준치",
        "stat_p80_m": "이 값 이상을 사용하는 임차인은 전체 상위 20%에 해당합니다. 업종 특성에 맞는지 확인하시기 바랍니다.",
        "stat_p20":   "하위 20% 기준치",
        "stat_p20_m": "이 값 이하를 사용하는 임차인은 전체 하위 20%에 해당합니다.",
        "stat_hi":    "증가 기준 (상위 {pct}%)",
        "stat_hi_m":  "사용량 증가폭이 이 값을 초과하는 임차인은 상위 {pct}% 증가 구간에 해당하며 검토 대상으로 분류됩니다.",
        "stat_lo":    "감소 기준 (하위 {pct}%)",
        "stat_lo_m":  "사용량 감소폭이 이 값을 초과하는 임차인은 하위 {pct}% 감소 구간에 해당합니다.",
        "charts_title":       "사용량 분포 및 변화 현황",
        "chart_hist_title":   "{name} — 이번 기간 사용량 분포",
        "chart_bar_title":    "{name} — 임차인별 사용량 변화",
        "chart_donut_title":  "임차인 상태 분포",
        "chart_hist_caption": (
            "그림: 전체 임차인의 {name} 사용량 분포. "
            "황색 막대 = 하위/상위 {pct}%. 빨간 점선 = 중앙값."
        ),
        "chart_bar_caption": (
            "그림: 임차인별 전기 대비 사용량 변화 (상태별 색상 구분). "
            "오른쪽 도넛 차트는 임차인 상태 비율을 나타냅니다."
        ),
        "flagged_title": "주의 임차인",
        "flagged_note":  "아래 임차인은 사용량 변화 또는 변화율이 하위/상위 {pct}% 기준을 벗어나 후속 조치가 필요합니다.",
        "legend_title":  "상태 안내",
        "all_title":     "전체 임차인",
        "th_no":     "번호",
        "th_tenant": "임차인",
        "th_bldg":   "건물",
        "th_floor":  "층",
        "th_last":   "이전({unit})",
        "th_this":   "이번({unit})",
        "th_change": "변화량({unit})",
        "th_pct":    "변화율 (%)",
        "th_status": "상태",
        "th_leg_status":    "상태",
        "th_leg_meaning":   "의미",
        "th_leg_condition": "판정 기준",
        "th_leg_action":    "권장 조치",
        "desc_Critical": "사용량 급증 및 절대량 높음",
        "desc_Watch":    "전체 사용량 높음, 변화율 보통",
        "desc_Alert":    "기준 대비 급격한 변화율 상승",
        "desc_Stable":   "사용량 낮고 감소 중",
        "desc_Normal":   "정상 범위 내",
        "desc_No Data":  "이전 기간 데이터 없음",
        # 평이한 판정 기준 설명 ({pct}는 렌더링 시 대입)
        "cond_Critical": "변화량(절대)과 변화율(%) 모두 상위 {pct}% 기준을 초과",
        "cond_Watch":    "변화량 또는 변화율 중 하나만 상위 {pct}% 기준 초과 (두 조건 동시 초과 시 과다로 분류)",
        "cond_Alert":    "변화량 또는 변화율 중 하나만 하위 {pct}% 기준 미만 (두 조건 동시 해당 시 안정으로 분류)",
        "cond_Stable":   "변화량(절대)과 변화율(%) 모두 하위 {pct}% 기준 미만",
        "cond_Normal":   "상위·하위 기준 어느 조건에도 해당 없는 중간 범위",
        "cond_No Data":  "이전 기간 검침값 없음 — 변화율 산출 불가",
        "leg_condition_note": (
            "기준값은 항목별로 독립 산출됩니다. "
            "이전·이번 기간 모두 검침 데이터가 있는 임차인만 포함합니다. "
            "상위 {pct}% 기준 = 해당 항목 변화량 분포의 상위 {pct}번째 백분위수 (변화율도 동일 방식 적용). "
            "하위 {pct}% 기준 = 하위 {pct}번째 백분위수."
        ),
        "act_Critical":  "즉시 점검 — 누수, 장비 이상 또는 비정상적 사용 여부 확인",
        "act_Watch":     "모니터링 — 사용량 높음; 업종 특성과 부합하는지 확인",
        "act_Alert":     "원인 확인 — 신규 장비, 직원 추가, 또는 검침 오류 가능성",
        "act_Stable":    "조치 불필요",
        "act_Normal":    "조치 불필요",
        "act_No Data":   "신규 임차인 또는 첫 청구 기간 여부 확인",
        "footer_left": "관리비 분석 보고서  ·  대외비",
        "footer_page": "{n} / {total} 페이지",
        "end_title":   "보고서 끝",
        "end_note":    (
            "작성일: {date}. 본 보고서는 내부 관리 목적으로만 사용하시기 바랍니다. "
            "모든 사용량 수치는 업로드된 파일의 검침 데이터를 기반으로 합니다."
        ),
        # executive summary
        "exec_title": "요약",
        "exec_all_clear": (
            "이번 기간 {n_util}개 항목, 전체 <b>{n_total}</b>명의 임차인이 정상 범위 내에 있습니다. "
            "즉각적인 조치가 필요한 사항은 없습니다."
        ),
        "exec_critical_lead": (
            "이번 기간 <b>{n_critical}</b>명의 임차인이 <b>즉각적인 확인</b>이 필요합니다."
        ),
        "exec_building_concentration": (
            " <b>{bldg}</b>동에서 가장 많은 이상 임차인({n}명)이 발생했습니다."
        ),
        "exec_vacancy": (
            " {n}개 호실이 이전 기간 데이터가 있으나 이번 기간 사용량이 0으로, 공실 가능성이 있습니다."
        ),
        "exec_new": (
            " {n}명의 신규 임차인이 첫 청구 기간에 있으며, 다음 기간부터 비교 분석이 가능합니다."
        ),
        "exec_watch_alert": (
            " 또한 주의 등급 {n_watch}명, 경보 등급 {n_alert}명이 있습니다."
        ),
        # action checklist
        "checklist_title":  "조치 체크리스트",
        "checklist_note":   (
            "이 페이지를 활용하여 이상 임차인에 대한 후속 조치를 관리하십시오. "
            "확인 및 해결 완료 시 해당 항목을 표시하십시오."
        ),
        "cl_th_no":      "번호",
        "cl_th_tenant":  "임차인",
        "cl_th_bldg":    "건물/층",
        "cl_th_utility": "항목",
        "cl_th_status":  "상태",
        "cl_th_issue":   "이슈",
        "cl_th_check":   "확인",
        "cl_th_resolve": "해결",
        "cl_th_remarks": "조치 비고",
        "cl_none":       "이번 기간 이상 임차인이 없습니다. 조치가 필요하지 않습니다.",
        "cl_issue_critical": "사용량 +{change} ({pct}) — 절대량 및 변화율 모두 높음",
        "cl_issue_alert":    "사용량 +{change} ({pct}) — 기준 대비 급격한 변화율 상승",
        "cl_issue_watch":    "사용량 +{change} ({pct}) — 전체 사용량 높음",
        # building totals
        "bldg_totals_title": "건물별 사용량",
        "col_bldg":   "건물",
        "col_curr_t": "이번 기간",
        "col_prev_t": "이전 기간",
        "col_chg_t":  "변화량",
        "col_pct_t":  "변화율 (%)",
        # data coverage
        "coverage_title": "데이터 커버리지",
        "col_utility_c":  "항목",
        "col_total_c":    "전체 임차인",
        "col_with_data":  "비교 가능",
        "col_no_data_c":  "이전 데이터 없음",
        "col_coverage":   "커버리지",
        # top 10 consumers
        "top10_title": "사용량 상위 10개 임차인",
        "top10_note":  "이번 기간 절대 사용량 기준 상위 10개 임차인입니다. 변화 상태와 무관하게 표시됩니다.",
        "th_rank":     "순위",
        "th_curr_use": "이번 사용량",
        # critical profile cards
        "profile_title": "과다 임차인 상세 현황",
        "profile_note":  (
            "과다 등급 임차인의 항목별 상태를 한눈에 확인할 수 있습니다. "
            "각 행은 해당 임차인의 모든 항목별 상태와 변화량을 나타냅니다."
        ),
        "profile_none":  "이번 기간 과다 임차인이 없습니다.",
        # seasonal note
        "seasonal_title":  "계절 참고사항",
        "seasonal_spring": (
            "이번 보고서는 봄철 전환 기간(3~5월)을 포함합니다. "
            "냉방 사용량 증가 및 난방 사용량 감소는 계절적 요인으로 예상되는 변화입니다."
        ),
        "seasonal_summer": (
            "이번 보고서는 여름철(6~8월)을 포함합니다. "
            "냉방 부하로 인한 전기 및 냉방 사용량 증가가 예상됩니다."
        ),
        "seasonal_fall": (
            "이번 보고서는 가을철 전환 기간(9~11월)을 포함합니다. "
            "난방 사용량이 증가하는 시기이므로, 열 사용량 급증은 계절적 요인을 먼저 확인하시기 바랍니다."
        ),
        "seasonal_winter": (
            "이번 보고서는 겨울철(12~2월)을 포함합니다. "
            "난방 사용량이 계절적으로 높은 시기입니다. "
            "전년 동기와 비교하는 것이 더 의미 있는 기준이 될 수 있습니다."
        ),
        # building comparison chart
        "bldg_chart_title":   "{name} — 건물별 평균 사용량",
        "bldg_chart_caption": "그림: 건물별 이번 기간 평균 사용량. 진한 막대 = 전체 평균 이상.",
        # critical alerts page
        "critical_title": "과다 알림",
        "critical_intro": (
            "아래 임차인은 <b>즉각적인 확인</b>이 필요합니다. "
            "사용량 절대 증가폭과 변화율 모두 상위 {pct}% 기준을 초과하여 "
            "이번 기간 비정상적인 소비 급증이 감지되었습니다."
        ),
        "critical_none": (
            "이번 기간 과다 알림이 없습니다. "
            "모든 임차인이 정상 사용 범위 내에 있습니다."
        ),
        "critical_section": "{name}  —  과다 임차인 {n}개",
        "critical_interp": (
            "전체 <b>{total}</b>개 임차인 중 <b>{n}개</b>가 {utility} 과다 등급으로 분류되었습니다. "
            "이들의 이번 기간 평균 사용량은 <b>{avg_crit}</b>으로, "
            "전체 평균 {avg_all} 대비 <b>{comparison}</b>. "
            "가장 높은 증가폭은 <b>{top_tenant}</b>으로 "
            "<b>{top_change} {unit}</b> ({top_pct}) 증가했습니다. "
            "즉각적인 점검이 권고됩니다."
        ),
        "critical_watch_note": (
            "또한 해당 항목에서 <b>주의</b> 등급 {n_watch}개, <b>경보</b> 등급 {n_alert}개 임차인이 있습니다. "
            "전체 목록은 상세 섹션을 참고하시기 바랍니다."
        ),
        # special analysis
        "section_special":   "  특별 분석  —  공실 및 이상 활동",
        "special_intro": (
            "이 섹션은 일반 과다/경보 기준 외에 추가 확인이 필요한 임차인을 정리합니다. "
            "공실 의심, 첫 청구 기간 신규 임차인, 복수 항목 동시 이상, 사용량 급격 감소 임차인을 포함합니다."
        ),
        "sub_vacancy":     "공실 의심 임차인",
        "vacancy_note":    "이전 기간 데이터가 있으나 이번 기간 사용량이 0 또는 근사 0인 임차인입니다. "
                           "공실 또는 폐업 가능성이 있습니다.",
        "vacancy_none":    "현재 사용량이 0인 임차인이 없습니다.",
        "sub_new_tenants": "신규 임차인 / 첫 청구 기간",
        "new_tenants_note":"이전 기간 데이터가 없는 임차인입니다(신규 입주 또는 첫 청구 기간). "
                           "'데이터 없음'으로 표시되며 변화 분석이 불가합니다.",
        "new_none":         "신규 임차인이 없습니다.",
        "sub_missing_data": "검침 데이터 누락",
        "missing_data_note":"이번 기간 검침값이 기록되지 않은 임차인입니다. "
                            "검침 누락 또는 비사용 공간 여부를 확인하십시오.",
        "missing_none":     "누락된 검침 데이터가 없습니다.",
        "sub_multi_flag":  "복수 항목 경보 임차인",
        "multi_flag_note": "두 개 이상의 항목(수도, 전기, 냉난방 등)에서 동시에 과다·주의·경보로 분류된 임차인입니다. "
                           "복수 항목 동시 이상은 설비 이상 등 구조적 문제를 시사할 수 있습니다.",
        "multi_flag_none": "복수 항목에서 동시에 이상이 감지된 임차인이 없습니다.",
        "sub_sharp_drop":  "급격한 사용량 감소 임차인",
        "sharp_drop_note": "두 개 이상의 항목에서 '안정' 등급(낮은 사용량 + 급격한 감소)으로 분류된 임차인입니다. "
                           "복수 항목 동시 감소는 공실, 폐업, 또는 계량기 이상을 의심해볼 수 있습니다.",
        "sharp_drop_none": "복수 항목에서 동시에 사용량이 급격히 감소한 임차인이 없습니다.",
        "th_utilities":    "해당 항목",
        # median vs mean interpretation
        "med_vs_avg_high": (
            "평균({avg})이 중앙값({med})보다 상당히 높습니다. "
            "일부 고사용 임차인이 평균을 끌어올리고 있으며, "
            "일반 임차인의 기준치로는 <b>중앙값</b>이 더 적합합니다."
        ),
        "med_vs_avg_low": (
            "평균({avg})이 중앙값({med})보다 낮습니다. "
            "사용량이 거의 없는 임차인들이 평균을 낮추고 있으므로 "
            "해당 호실의 공실 여부를 확인하시기 바랍니다."
        ),
        "med_vs_avg_even": (
            "평균({avg})과 중앙값({med})이 유사합니다. "
            "전체 사용량 분포가 비교적 고르며 극단적인 이상치가 없는 상태입니다."
        ),
    },
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify(ch, pt, hi_c, lo_c, hi_p, lo_p):
    if pd.isna(ch) or pd.isna(pt):
        return "No Data"
    if ch >= hi_c and pt >= hi_p:
        return "Critical"
    if ch >= hi_c and pt < hi_p:
        return "Watch"
    if ch <= lo_c and pt >= hi_p:
        return "Alert"
    if ch <= lo_c and pt <= lo_p:
        return "Stable"
    return "Normal"


def _fmt(val, unit="", sign=False):
    if pd.isna(val):
        return "—"
    s = f"{val:+,.2f}" if sign else f"{val:,.2f}"
    return f"{s} {unit}".strip() if unit else s


def _pct(val):
    return f"{val:+.1f}%" if not pd.isna(val) else "—"

def _pct_val(val):
    """Numeric-only percent for table cells (unit shown in header)."""
    return f"{val:+.1f}" if not pd.isna(val) else "—"


def _png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close(fig)
    return buf


# ── Chart generators ─────────────────────────────────────────────────────────

def _v(r):
    """Safe float for a row's change value — NaN → 0."""
    v = r.get("change")
    return 0.0 if (v is None or pd.isna(v)) else float(v)


def _chart_change_bar(rows, unit, title, status_labels, max_rows=40):
    rows = [r for r in rows if r.get("status") != "No Data"]
    rows = sorted(rows, key=_v)
    if len(rows) > max_rows:
        half = max_rows // 2
        rows = rows[:half] + rows[-half:]

    labels  = [textwrap.shorten(str(r.get("brand", "")), width=28, placeholder="…") for r in rows]
    values  = [_v(r) for r in rows]
    mcolors = [STATUS_COLOR_M.get(r.get("status", "Normal"), M_NORMAL) for r in rows]

    n = len(rows)
    fig_h = max(4.0, n * 0.38 + 1.0)
    fig, ax = plt.subplots(figsize=(11.0, fig_h), facecolor="white")

    ax.barh(range(n), values, color=mcolors,
            edgecolor="white", linewidth=0.5, height=0.72)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=10)
    ax.axvline(0, color="#333333", linewidth=1.2)
    ax.set_xlabel(f"({unit})", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("data", -0.5))
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.set_facecolor("white")

    # Value labels
    xlim = ax.get_xlim()
    x_off = 0.01 * (xlim[1] - xlim[0])
    for i, v in enumerate(values):
        if abs(v) > 0:
            ha = "left" if v >= 0 else "right"
            ax.text(v + (x_off if v >= 0 else -x_off), i, f"{v:+,.1f}",
                    va="center", ha=ha, fontsize=9, color="#333333")

    # Legend with translated labels
    seen = {}
    for r in rows:
        key = r.get("status", "Normal")
        if key not in seen:
            label = status_labels.get(key, key)
            seen[key] = mpatches.Patch(color=STATUS_COLOR_M.get(key, M_NORMAL), label=label)
    if seen:
        ax.legend(handles=list(seen.values()), fontsize=9,
                  loc="lower right", framealpha=0.9)

    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_histogram(values, hi, lo, unit, title, xlabel_suffix=""):
    vals = np.array([float(v) for v in values if v is not None and not pd.isna(v)], dtype=float)
    if len(vals) == 0:
        return None

    fig, ax = plt.subplots(figsize=(8.5, 3.5), facecolor="white")
    n_bins = 80
    counts, edges = np.histogram(vals, bins=n_bins)
    mids   = (edges[:-1] + edges[1:]) / 2
    widths = edges[1:] - edges[:-1]
    tail_mask = (mids <= lo) | (mids >= hi)

    bar_colors = [M_WATCH if tail_mask[i] else M_BAR for i in range(len(mids))]
    ax.bar(mids, counts, width=widths * 0.92, color=bar_colors,
           edgecolor="white", linewidth=0.5)

    med = float(np.median(vals))

    # Tail shading
    xmin, xmax = float(vals.min()), float(vals.max())
    if lo > xmin:
        ax.axvspan(xmin, lo, color=M_WATCH, alpha=0.10, linewidth=0)
        ax.axvline(lo, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7)
    if hi < xmax:
        ax.axvspan(hi, xmax, color=M_WATCH, alpha=0.10, linewidth=0)
        ax.axvline(hi, color="#555555", linewidth=1.2, linestyle="--", alpha=0.7,
                   label=f"Threshold: {lo:,.1f} / {hi:,.1f}")

    ax.axvline(med, color="#C44E52", linewidth=1.5, linestyle="--")

    ax.set_xlabel(f"({unit})", fontsize=9)
    ax.set_ylabel(xlabel_suffix, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold", color="#1B2A3B", pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(0, counts.max() * 1.15)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5, linestyle="--")
    ax.legend(handles=[
        _Line2D([0], [0], color="#C44E52", linewidth=1.5, linestyle="--", label=f"Median: {med:,.2f} {unit}"),
    ], fontsize=8, framealpha=0.9)
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _chart_status_donut(status_counts, title, status_labels):
    order  = ["Critical", "Watch", "Alert", "Normal", "Stable", "No Data"]
    keys   = [s for s in order if status_counts.get(s, 0) > 0]
    sizes  = [status_counts[s] for s in keys]
    clrs   = [STATUS_COLOR_M.get(s, M_NORMAL) for s in keys]
    labels = [status_labels.get(s, s) for s in keys]

    if not sizes or sum(sizes) == 0:
        return None

    fig, ax = plt.subplots(figsize=(7.0, 4.5), facecolor="white")
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=clrs, autopct="%1.0f%%",
        pctdistance=0.72, startangle=90,
        wedgeprops=dict(width=0.48, edgecolor="white", linewidth=1.5),
    )
    LIGHT_WEDGE = {M_ALERT, M_NODATA}
    for at, clr in zip(autotexts, clrs):
        at.set_fontsize(10)
        at.set_color("#333333" if clr in LIGHT_WEDGE else "white")

    ax.legend(
        wedges, [f"{l} ({n})" for l, n in zip(labels, sizes)],
        fontsize=10, loc="center left", bbox_to_anchor=(0.88, 0.5),
        framealpha=0.9,
    )
    ax.set_title(title, fontsize=12, fontweight="bold", color="#1B2A3B")
    fig.tight_layout(pad=0.5)
    return _png(fig)


# ── ReportLab styles ─────────────────────────────────────────────────────────

def _make_styles():
    s = {}
    s["cover_title"] = ParagraphStyle(
        "CoverTitle", fontSize=26, fontName="NanumGothic-Bold",
        leading=34, textColor=C_NAVY, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=10,
    )
    s["cover_sub"] = ParagraphStyle(
        "CoverSub", fontSize=13, fontName="NanumGothic",
        leading=18, textColor=C_SUBTEXT, alignment=TA_LEFT,
        spaceBefore=0, spaceAfter=8,
    )
    s["section_title"] = ParagraphStyle(
        "SectionTitle", fontSize=14, fontName="NanumGothic-Bold",
        textColor=C_WHITE, alignment=TA_LEFT,
        leftIndent=6, spaceBefore=2, spaceAfter=2,
    )
    s["sub_title"] = ParagraphStyle(
        "SubTitle", fontSize=11, fontName="NanumGothic-Bold",
        textColor=C_NAVY, alignment=TA_LEFT,
        spaceBefore=10, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "Body", fontSize=9, fontName="NanumGothic",
        textColor=C_TEXT, alignment=TA_LEFT,
        spaceAfter=3, leading=14,
    )
    s["caption"] = ParagraphStyle(
        "Caption", fontSize=8, fontName="NanumGothic",
        textColor=C_SUBTEXT, alignment=TA_CENTER, spaceAfter=2,
    )
    s["note"] = ParagraphStyle(
        "Note", fontSize=8, fontName="NanumGothic",
        textColor=C_SUBTEXT, alignment=TA_LEFT, spaceAfter=4, leading=13,
    )
    s["table_hdr"] = ParagraphStyle(
        "TableHdr", fontSize=8, fontName="NanumGothic-Bold",
        textColor=C_WHITE, alignment=TA_LEFT,
    )
    s["table_subhdr"] = ParagraphStyle(
        "TableSubHdr", fontSize=8, fontName="NanumGothic-Bold",
        textColor=C_NAVY, alignment=TA_CENTER,
    )
    s["table_cell"] = ParagraphStyle(
        "TableCell", fontSize=8, fontName="NanumGothic",
        textColor=C_TEXT, alignment=TA_LEFT, leading=12,
    )
    s["table_cell_c"] = ParagraphStyle(
        "TableCellC", fontSize=8, fontName="NanumGothic",
        textColor=C_TEXT, alignment=TA_CENTER, leading=12,
    )
    return s


# ── Layout helpers ────────────────────────────────────────────────────────────

def _section_bar(text, styles, inner_w):
    return Table(
        [[Paragraph(text, styles["section_title"])]],
        colWidths=[inner_w],
        style=TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_NAVY),
            ("TOPPADDING",    (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ]),
    )


def _stats_table(stat_rows, col_widths, styles):
    data = [[
        Paragraph(stat_rows[0][0], styles["table_hdr"]),
        Paragraph(stat_rows[0][1], styles["table_hdr"]),
        Paragraph(stat_rows[0][2], styles["table_hdr"]),
    ]]
    ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("GRID",           (0, 0), (-1, -1), 0.4, C_DIVIDER),
    ])
    for metric, value, meaning in stat_rows[1:]:
        data.append([
            Paragraph(metric,  styles["table_cell"]),
            Paragraph(value,   styles["table_cell"]),
            Paragraph(meaning, styles["table_cell"]),
        ])
    return Table(data, colWidths=col_widths, style=ts, repeatRows=1)


def _detail_table(rows, unit, col_widths, styles, T):
    # Column order: 상태 first so status color is a left-margin indicator
    headers = [
        T["th_status"],
        T["th_no"],
        T["th_tenant"], T["th_bldg"], T["th_floor"],
        T["th_last"].format(unit=unit), T["th_this"].format(unit=unit),
        T["th_change"].format(unit=unit), T["th_pct"],
    ]
    data = [[Paragraph(h, styles["table_hdr"]) for h in headers]]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("ROWBACKGROUNDS",(1, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        # 번호 column: tight padding so the number stays on one line
        ("LEFTPADDING",   (1, 0), (1, -1), 2),
        ("RIGHTPADDING",  (1, 0), (1, -1), 2),
        ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        # numeric columns right-aligned
        ("ALIGN",         (5, 0), (8, -1), "RIGHT"),
    ])
    for i, r in enumerate(rows, 1):
        status = r.get("status", "Normal")
        rc = STATUS_COLOR_RL.get(status, C_NORMAL)
        data.append([
            Paragraph(T.get(status, status),                            styles["table_cell_c"]),
            Paragraph(str(i),                                           styles["table_cell_c"]),
            Paragraph(textwrap.shorten(r.get("brand", ""), 28, placeholder="…"), styles["table_cell"]),
            Paragraph(r.get("building", ""),                            styles["table_cell_c"]),
            Paragraph(r.get("floor", ""),                               styles["table_cell_c"]),
            Paragraph(_fmt(r.get("prev")),                              styles["table_cell_c"]),
            Paragraph(_fmt(r.get("curr")),                              styles["table_cell_c"]),
            Paragraph(_fmt(r.get("change"), sign=True),                 styles["table_cell_c"]),
            Paragraph(_pct_val(r.get("pct")),                           styles["table_cell_c"]),
        ])
        # highlight only the 상태 cell (column 0)
        ts.add("BACKGROUND", (0, i), (0, i), rc)
        if status == "Critical":
            ts.add("TEXTCOLOR", (0, i), (0, i), colors.white)
    return Table(data, colWidths=col_widths, style=ts, repeatRows=1)


def _img_flowable(png_buf, width_cm, styles, caption=""):
    pil = PILImage.open(png_buf)
    w_px, h_px = pil.size
    png_buf.seek(0)
    items = [Image(png_buf, width=width_cm * cm, height=width_cm * cm * h_px / w_px)]
    if caption:
        items.append(Paragraph(caption, styles["caption"]))
    return items


def _critical_alerts_page(story, util_data, T, styles, content_w, tail_pct):
    """Append a Critical Alerts summary page to the story."""
    story.append(_section_bar(f"  {T['critical_title']}", styles, content_w))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(
        T["critical_intro"].format(pct=tail_pct), styles["body"],
    ))
    story.append(Spacer(1, 0.4 * cm))

    any_critical = any(
        ud["status_counts"].get("Critical", 0) > 0 for ud in util_data.values()
    )

    if not any_critical:
        story.append(Paragraph(T["critical_none"], styles["body"]))
        story.append(PageBreak())
        return

    _fixed_cw = (0.9 + 1.1 + 1.2 + 2.0 + 2.0 + 2.0 + 1.4) * cm
    cw = [0.9*cm, content_w - _fixed_cw, 1.1*cm, 1.2*cm, 2.0*cm, 2.0*cm, 2.0*cm, 1.4*cm]

    for prefix, ud in util_data.items():
        crit_rows = [r for r in ud["rows"] if r["status"] == "Critical"]
        if not crit_rows:
            continue

        unit     = ud["unit"]
        name     = ud["name"]
        sc       = ud["status_counts"]
        n_crit   = len(crit_rows)
        n_total  = ud["n_data"]
        s_cu     = ud["s_cu"]
        all_avg  = float(s_cu.mean()) if not s_cu.empty else float("nan")

        # Avg usage of critical tenants this period
        crit_curr = [r["curr"] for r in crit_rows if not pd.isna(r.get("curr"))]
        avg_crit  = float(np.mean(crit_curr)) if crit_curr else float("nan")
        pct_above = round((avg_crit / all_avg - 1) * 100) if all_avg and not np.isnan(avg_crit) else 0

        # "X% above / below" phrasing
        if lang_hint := T.get("footer_left", ""):
            _ko = "대외비" in lang_hint
        else:
            _ko = False
        if pct_above >= 0:
            comparison = (f"{pct_above}% 높습니다" if _ko else f"+{pct_above}% above")
        else:
            comparison = (f"{abs(pct_above)}% 낮습니다" if _ko else f"{abs(pct_above)}% below")

        # Worst offender by absolute change
        top = max(crit_rows, key=lambda r: r.get("change") or 0)

        # Critical tenant table (no status column — all are Critical)
        hdr = [T["th_no"], T["th_tenant"], T["th_bldg"], T["th_floor"],
               T["th_last"].format(unit=unit), T["th_this"].format(unit=unit),
               T["th_change"].format(unit=unit), T["th_pct"]]
        data = [[Paragraph(h, styles["table_hdr"]) for h in hdr]]
        ts = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  C_CRITICAL),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#FFF0F0"), colors.HexColor("#FFE0E0")]),
            # 번호 column — tight padding to keep number horizontal
            ("LEFTPADDING",   (0, 1), (0, -1), 2),
            ("RIGHTPADDING",  (0, 1), (0, -1), 2),
            ("ALIGN",         (0, 0), (0, -1), "CENTER"),
        ])
        for i, r in enumerate(crit_rows, 1):
            data.append([
                Paragraph(str(i),                                               styles["table_cell_c"]),
                Paragraph(textwrap.shorten(r.get("brand",""), 28, placeholder="…"), styles["table_cell"]),
                Paragraph(r.get("building", ""),                                styles["table_cell_c"]),
                Paragraph(r.get("floor",    ""),                                styles["table_cell_c"]),
                Paragraph(_fmt(r.get("prev")),                                  styles["table_cell_c"]),
                Paragraph(_fmt(r.get("curr")),                                  styles["table_cell_c"]),
                Paragraph(_fmt(r.get("change"), sign=True),                     styles["table_cell_c"]),
                Paragraph(_pct_val(r.get("pct")),                               styles["table_cell_c"]),
            ])

        # Keep sub-header + interpretation + table together
        section_block = [
            Paragraph(T["critical_section"].format(name=name, n=n_crit), styles["sub_title"]),
            Paragraph(
                T["critical_interp"].format(
                    total      = n_total,
                    n          = n_crit,
                    utility    = name,
                    avg_crit   = _fmt(avg_crit, unit),
                    avg_all    = _fmt(all_avg,  unit),
                    comparison = comparison,
                    top_tenant = textwrap.shorten(top.get("brand", "—"), 30, placeholder="…"),
                    top_change = _fmt(top.get("change"), sign=True),
                    top_pct    = _pct(top.get("pct")),
                    unit       = unit,
                ),
                styles["body"],
            ),
            Spacer(1, 0.25 * cm),
            Table(data, colWidths=cw, style=ts, repeatRows=1),
        ]

        # Watch/Alert footnote
        n_watch = sc.get("Watch", 0)
        n_alert = sc.get("Alert", 0)
        if n_watch + n_alert > 0:
            section_block += [
                Spacer(1, 0.15 * cm),
                Paragraph(T["critical_watch_note"].format(n_watch=n_watch, n_alert=n_alert), styles["note"]),
            ]

        story.append(KeepTogether(section_block))
        story.append(Spacer(1, 0.5 * cm))

    story.append(PageBreak())


def _vacancy_section(story, util_data, T, styles, content_w):
    """Cross-utility: vacancy, new tenants, multi-flag, sharp drop."""
    story.append(_section_bar(T["section_special"], styles, content_w))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(T["special_intro"], styles["note"]))
    story.append(Spacer(1, 0.5 * cm))

    # Collect per-(brand, building) status across utilities
    all_brands = {}
    for prefix, ud in util_data.items():
        meta = ud["meta"]
        for r in ud["rows"]:
            key = (r["brand"], r["building"])
            if key not in all_brands:
                all_brands[key] = {}
            all_brands[key][prefix] = {**r, "meta": meta}

    def _mini_table(rows_data, col_widths, styles_arg):
        """Simple table with alternating row colors."""
        ts = TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
            ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",    (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS",(0, 1), (-1, -1),
             [colors.HexColor("#F5F7FA"), colors.white]),
        ])
        return Table(rows_data, colWidths=col_widths, style=ts)

    def _util_names(prefix_list, lang_key):
        names = []
        for p in prefix_list:
            m = util_data[p]["meta"]
            names.append(m.get(lang_key, m.get("en", p)))
        return ", ".join(names)

    lang_key = "ko" if T.get("section_special", "").startswith("  특별") else "en"

    # ── 1. Suspected vacant ──────────────────────────────────────────────
    vac_rows = [[
        Paragraph(T["th_tenant"], styles["table_hdr"]),
        Paragraph(T["th_bldg"],   styles["table_hdr"]),
        Paragraph(T["th_utilities"], styles["table_hdr"]),
    ]]
    for (brand, bldg), util_rows in sorted(all_brands.items()):
        zero_utils = [
            p for p, r in util_rows.items()
            if r.get("curr") is not None
            and not pd.isna(r.get("curr", np.nan))
            and float(r["curr"]) < 0.01
            and r.get("prev") is not None
            and not pd.isna(r.get("prev", np.nan))
        ]
        if zero_utils:
            vac_rows.append([
                Paragraph(str(brand), styles["table_cell"]),
                Paragraph(str(bldg),  styles["table_cell"]),
                Paragraph(_util_names(zero_utils, lang_key), styles["table_cell"]),
            ])
    vac_block = [
        Paragraph(T["sub_vacancy"], styles["sub_title"]),
        Paragraph(T["vacancy_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Paragraph(T["vacancy_none"], styles["note"]) if len(vac_rows) == 1
        else _mini_table(vac_rows, [5.5*cm, 1.8*cm, content_w - 7.3*cm], styles),
    ]
    story.append(KeepTogether(vac_block))
    story.append(Spacer(1, 0.6 * cm))

    # ── 2a. New tenants (first reading) ──────────────────────────────────
    new_rows = [[
        Paragraph(T["th_tenant"],    styles["table_hdr"]),
        Paragraph(T["th_bldg"],      styles["table_hdr"]),
        Paragraph(T["th_utilities"], styles["table_hdr"]),
    ]]
    for (brand, bldg), util_rows in sorted(all_brands.items()):
        new_utils = [
            p for p, r in util_rows.items()
            if r.get("status") == "No Data"
            and (r.get("prev") is None or pd.isna(r.get("prev", np.nan)))
            and r.get("curr") is not None
            and not pd.isna(r.get("curr", np.nan))
            and float(r["curr"]) > 0.01
        ]
        if new_utils:
            new_rows.append([
                Paragraph(str(brand), styles["table_cell"]),
                Paragraph(str(bldg),  styles["table_cell"]),
                Paragraph(_util_names(new_utils, lang_key), styles["table_cell"]),
            ])
    new_block = [
        Paragraph(T["sub_new_tenants"], styles["sub_title"]),
        Paragraph(T["new_tenants_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Paragraph(T["new_none"], styles["note"]) if len(new_rows) == 1
        else _mini_table(new_rows, [5.5*cm, 1.8*cm, content_w - 7.3*cm], styles),
    ]
    story.append(KeepTogether(new_block))
    story.append(Spacer(1, 0.5 * cm))

    # ── 2b. Missing data (no readings at all) ────────────────────────────
    miss_rows = [[
        Paragraph(T["th_tenant"],    styles["table_hdr"]),
        Paragraph(T["th_bldg"],      styles["table_hdr"]),
        Paragraph(T["th_utilities"], styles["table_hdr"]),
    ]]
    for (brand, bldg), util_rows in sorted(all_brands.items()):
        miss_utils = [
            p for p, r in util_rows.items()
            if (r.get("curr") is None or pd.isna(r.get("curr", np.nan)))
        ]
        if miss_utils:
            miss_rows.append([
                Paragraph(str(brand), styles["table_cell"]),
                Paragraph(str(bldg),  styles["table_cell"]),
                Paragraph(_util_names(miss_utils, lang_key), styles["table_cell"]),
            ])
    miss_block = [
        Paragraph(T["sub_missing_data"], styles["sub_title"]),
        Paragraph(T["missing_data_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Paragraph(T["missing_none"], styles["note"]) if len(miss_rows) == 1
        else _mini_table(miss_rows, [5.5*cm, 1.8*cm, content_w - 7.3*cm], styles),
    ]
    story.append(KeepTogether(miss_block))
    story.append(Spacer(1, 0.6 * cm))

    # ── 3. Multi-utility alerts ──────────────────────────────────────────
    mf_rows = [[
        Paragraph(T["th_tenant"],    styles["table_hdr"]),
        Paragraph(T["th_bldg"],      styles["table_hdr"]),
        Paragraph(T["th_utilities"], styles["table_hdr"]),
        Paragraph(T["th_status"],    styles["table_hdr"]),
    ]]
    for (brand, bldg), util_rows in sorted(all_brands.items()):
        flagged = [
            (p, r["status"]) for p, r in util_rows.items()
            if r.get("status") in ("Critical", "Watch", "Alert")
        ]
        if len(flagged) >= 2:
            util_str = ", ".join(
                f"{_util_names([p], lang_key)} ({st})" for p, st in flagged
            )
            statuses = "/".join(dict.fromkeys(st for _, st in flagged))
            mf_rows.append([
                Paragraph(str(brand), styles["table_cell"]),
                Paragraph(str(bldg),  styles["table_cell"]),
                Paragraph(util_str,   styles["table_cell"]),
                Paragraph(statuses,   styles["table_cell"]),
            ])
    cws_mf = [5.0*cm, 1.8*cm, content_w - 9.8*cm, 3.0*cm]
    mf_block = [
        Paragraph(T["sub_multi_flag"], styles["sub_title"]),
        Paragraph(T["multi_flag_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Paragraph(T["multi_flag_none"], styles["note"]) if len(mf_rows) == 1
        else _mini_table(mf_rows, cws_mf, styles),
    ]
    story.append(KeepTogether(mf_block))
    story.append(Spacer(1, 0.6 * cm))

    # ── 4. Sharp drop ────────────────────────────────────────────────────
    sd_rows = [[
        Paragraph(T["th_tenant"],    styles["table_hdr"]),
        Paragraph(T["th_bldg"],      styles["table_hdr"]),
        Paragraph(T["th_utilities"], styles["table_hdr"]),
    ]]
    for (brand, bldg), util_rows in sorted(all_brands.items()):
        stable_utils = [p for p, r in util_rows.items() if r.get("status") == "Stable"]
        if len(stable_utils) >= 2:
            sd_rows.append([
                Paragraph(str(brand), styles["table_cell"]),
                Paragraph(str(bldg),  styles["table_cell"]),
                Paragraph(_util_names(stable_utils, lang_key), styles["table_cell"]),
            ])
    sd_block = [
        Paragraph(T["sub_sharp_drop"], styles["sub_title"]),
        Paragraph(T["sharp_drop_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Paragraph(T["sharp_drop_none"], styles["note"]) if len(sd_rows) == 1
        else _mini_table(sd_rows, [5.5*cm, 1.8*cm, content_w - 7.3*cm], styles),
    ]
    story.append(KeepTogether(sd_block))

    story.append(PageBreak())


def _seasonal_note(period_str, T):
    """Return a seasonal context string based on the billing period month, or None.
    Accepts '2024년 3월', '2024-03-01', or plain month integer."""
    import re as _re_s
    try:
        m = _re_s.search(r'(\d{1,2})월', str(period_str))
        if m:
            month = int(m.group(1))
        else:
            month = int(str(period_str).split("-")[1])
    except Exception:
        return None
    if month in (3, 4, 5):
        return T["seasonal_spring"]
    if month in (6, 7, 8):
        return T["seasonal_summer"]
    if month in (9, 10, 11):
        return T["seasonal_fall"]
    return T["seasonal_winter"]


def _building_totals_table(util_data, T, styles, content_w):
    """Table: per-building totals for each utility.

    Layout (two header rows):
      Row 0 – Building (rowspan 2) | Utility name spanning 3 cols | ...
      Row 1 – (spanned)            | Curr | Chg | %  | Curr | Chg | % | ...
    """
    all_bldgs = sorted({r["building"] for ud in util_data.values() for r in ud["rows"] if r.get("building")})
    if not all_bldgs:
        return None

    prefixes = list(util_data.keys())
    lang_key = "ko" if T.get("col_bldg") == "건물" else "en"

    _UTIL_COLORS = {
        "water":  colors.HexColor("#2980B9"),
        "hwater": colors.HexColor("#E67E22"),
        "elect":  colors.HexColor("#27AE60"),
        "heat":   colors.HexColor("#E74C3C"),
    }
    _UTIL_TINT_COLORS = {
        "water":  colors.HexColor("#D6EAF8"),
        "hwater": colors.HexColor("#FDEBD0"),
        "elect":  colors.HexColor("#D5F5E3"),
        "heat":   colors.HexColor("#FADBD8"),
    }
    _UTIL_TINT_TEXT = {
        "water":  colors.HexColor("#1A5276"),
        "hwater": colors.HexColor("#784212"),
        "elect":  colors.HexColor("#1E8449"),
        "heat":   colors.HexColor("#922B21"),
    }

    # ── Header row 0: Building (rowspan 2) | util name spanning 3 cols | … ──
    hdr0 = [Paragraph(T["col_bldg"], styles["table_hdr"])]
    for p in prefixes:
        name = util_data[p]["meta"].get(lang_key, p)
        unit = util_data[p]["unit"]
        hdr0 += [Paragraph(f"{name}\n({unit})", styles["table_hdr"]), "", ""]

    # ── Header row 1: compact sub-labels (no spaces → no wrapping) ──
    ko = (lang_key == "ko")
    sub_labels = (
        ["이번기간", "변화량", "변화율(%)"] if ko
        else ["Current", "Chg", "Chg%"]
    )
    hdr1 = [Paragraph("", styles["table_hdr"])]
    for p in prefixes:
        sub_style = ParagraphStyle(
            f"subhdr_{p}",
            fontSize=7, fontName="NanumGothic-Bold",
            textColor=_UTIL_TINT_TEXT[p], alignment=TA_CENTER,
        )
        hdr1 += [Paragraph(lbl, sub_style) for lbl in sub_labels]

    data = [hdr0, hdr1]

    # ── Data rows ──
    pct_vals = {}
    for ri, bldg in enumerate(all_bldgs, 2):
        row = [Paragraph(str(bldg), styles["table_cell"])]
        for ci, p in enumerate(prefixes):
            ud = util_data[p]
            b_rows = [r for r in ud["rows"] if r.get("building") == bldg]
            curr_total = sum(float(r["curr"]) for r in b_rows
                             if r.get("curr") is not None and not pd.isna(r.get("curr", np.nan)))
            prev_total = sum(float(r["prev"]) for r in b_rows
                             if r.get("prev") is not None and not pd.isna(r.get("prev", np.nan)))
            chg = curr_total - prev_total
            pct = (chg / prev_total * 100) if prev_total > 0 else np.nan
            pct_col = 1 + ci * 3 + 2
            pct_vals[(ri, pct_col)] = pct
            row += [
                Paragraph(_fmt(curr_total), styles["table_cell_c"]),
                Paragraph(f"{chg:+,.1f}" if not pd.isna(chg) else "—", styles["table_cell_c"]),
                Paragraph(_pct_val(pct), styles["table_cell_c"]),
            ]
        data.append(row)

    # ── Column widths ──
    bldg_w = 1.0 * cm
    util_w = (content_w - bldg_w) / (len(prefixes) * 3)
    col_w  = [bldg_w] + [util_w] * (len(prefixes) * 3)

    ts = TableStyle([
        ("FONTSIZE",      (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 2),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 2),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        # Building column (spans both header rows)
        ("BACKGROUND",    (0, 0), (0, -1),  C_NAVY),
        ("FONTNAME",      (0, 0), (0, -1),  "NanumGothic-Bold"),
        ("ALIGN",         (0, 0), (0, -1),  "CENTER"),
        ("SPAN",          (0, 0), (0, 1)),
        # Data rows alternating
        ("ROWBACKGROUNDS",(0, 2), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("ALIGN",         (1, 2), (-1, -1), "RIGHT"),
        # Thick vertical dividers between utility groups
        *[("LINEAFTER", (1 + ci * 3 + 2, 0), (1 + ci * 3 + 2, -1), 1.2, C_NAVY)
          for ci in range(len(prefixes))],
    ])

    # Per-utility colors for row 0 (saturated) and row 1 (tint)
    for ci, p in enumerate(prefixes):
        col_start = 1 + ci * 3
        col_end   = col_start + 2
        ts.add("SPAN",       (col_start, 0), (col_end, 0))
        ts.add("BACKGROUND", (col_start, 0), (col_end, 0), _UTIL_COLORS[p])
        ts.add("ALIGN",      (col_start, 0), (col_end, 0), "CENTER")
        ts.add("BACKGROUND", (col_start, 1), (col_end, 1), _UTIL_TINT_COLORS[p])

    # % change cell coloring
    for (ri, pct_col), pct_v in pct_vals.items():
        if pd.isna(pct_v):
            continue
        if pct_v > 10:
            ts.add("TEXTCOLOR", (pct_col, ri), (pct_col, ri), C_CRITICAL)
            ts.add("FONTNAME",  (pct_col, ri), (pct_col, ri), "NanumGothic-Bold")
        elif pct_v < -10:
            ts.add("TEXTCOLOR", (pct_col, ri), (pct_col, ri), C_STABLE)
            ts.add("FONTNAME",  (pct_col, ri), (pct_col, ri), "NanumGothic-Bold")

    return Table(data, colWidths=col_w, style=ts, repeatRows=2)


def _data_coverage_table(util_data, T, styles, content_w):
    """Small table showing data availability per utility."""
    hdr = [T["col_utility_c"], T["col_total_c"], T["col_with_data"],
           T["col_no_data_c"], T["col_coverage"]]
    data = [[Paragraph(h, styles["table_hdr"]) for h in hdr]]
    for prefix, ud in util_data.items():
        n_total   = len(ud["rows"])
        n_nodata  = ud["status_counts"].get("No Data", 0)
        n_with    = n_total - n_nodata
        coverage  = f"{100 * n_with / n_total:.0f}%" if n_total > 0 else "—"
        lang_key = "ko" if T.get("col_bldg") == "건물" else "en"
        name = ud["meta"].get(lang_key, ud["meta"].get("en", prefix))
        data.append([
            Paragraph(name,             styles["table_cell"]),
            Paragraph(str(n_total),     styles["table_cell_c"]),
            Paragraph(str(n_with),      styles["table_cell_c"]),
            Paragraph(str(n_nodata),    styles["table_cell_c"]),
            Paragraph(coverage,         styles["table_cell_c"]),
        ])
    col_w = [4.0*cm, 2.5*cm, 2.5*cm, 3.0*cm, 2.5*cm]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
    ])
    return Table(data, colWidths=col_w, style=ts, repeatRows=1)


def _chart_building_comparison(util_data, T, prefix):
    """Horizontal bar chart: average current usage per building for one utility."""
    ud = util_data[prefix]
    bldg_data = {}
    for r in ud["rows"]:
        b = r.get("building", "")
        cu = r.get("curr")
        if b and cu is not None and not pd.isna(cu):
            bldg_data.setdefault(b, []).append(float(cu))
    if not bldg_data:
        return None

    bldgs = sorted(bldg_data.keys())
    avgs  = [np.mean(bldg_data[b]) for b in bldgs]
    overall_avg = np.mean(avgs)

    bar_colors = [M_CRITICAL if a > overall_avg * 1.1 else M_BAR for a in avgs]

    fig, ax = plt.subplots(figsize=(7.0, max(2.5, len(bldgs) * 0.55 + 0.8)), facecolor="white")
    ax.barh(bldgs, avgs, color=bar_colors, edgecolor="white", linewidth=0.5, height=0.6)
    ax.axvline(overall_avg, color="#555555", linewidth=1.2, linestyle="--",
               label=f"Avg: {overall_avg:,.1f}")
    xlim = ax.get_xlim()
    x_off = 0.01 * (xlim[1] - xlim[0])
    for i, (b, v) in enumerate(zip(bldgs, avgs)):
        ax.text(v + x_off, i, f"{v:,.1f}", va="center", ha="left", fontsize=9, color="#333333")
    ax.set_xlabel(f"({ud['unit']})", fontsize=10)
    ax.set_title(T["bldg_chart_title"].format(name=ud["name"]), fontsize=11, fontweight="bold",
                 color="#1B2A3B", pad=6)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10, length=0)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_facecolor("white")
    ax.set_xlim(0, max(avgs) * 1.2)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.5, linestyle="--")
    fig.tight_layout(pad=0.8)
    return _png(fig)


def _top10_section(story, rows, unit, name, T, styles, content_w, prepend=None):
    """Append Top 10 highest current usage table to story."""
    valid = [r for r in rows if r.get("curr") is not None and not pd.isna(r.get("curr", np.nan))]
    if not valid:
        return
    top10 = sorted(valid, key=lambda r: float(r["curr"]), reverse=True)[:10]

    hdr = [T["th_rank"], T["th_tenant"], T["th_bldg"], T["th_floor"],
           f"{T['th_curr_use']} ({unit})", T["th_pct"]]
    data = [[Paragraph(h, styles["table_hdr"]) for h in hdr]]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("ALIGN",         (4, 0), (5, -1),  "RIGHT"),
    ])
    for i, r in enumerate(top10, 1):
        status = r.get("status", "Normal")
        data.append([
            Paragraph(str(i),                                              styles["table_cell_c"]),
            Paragraph(textwrap.shorten(r.get("brand",""), 22, placeholder="…"), styles["table_cell"]),
            Paragraph(str(r.get("building", "")),                          styles["table_cell_c"]),
            Paragraph(str(r.get("floor", "")),                             styles["table_cell_c"]),
            Paragraph(_fmt(float(r["curr"])),                              styles["table_cell_c"]),
            Paragraph(_pct_val(r.get("pct")),                              styles["table_cell_c"]),
        ])
        if status in STATUS_COLOR_RL:
            ts.add("BACKGROUND", (0, i), (0, i), STATUS_COLOR_RL[status])
            if status == "Critical":
                ts.add("TEXTCOLOR", (0, i), (0, i), colors.white)

    _fixed_top10 = (1.0 + 1.5 + 1.5 + 3.0 + 2.0) * cm
    col_w = [1.0*cm, content_w - _fixed_top10, 1.5*cm, 1.5*cm, 3.0*cm, 2.0*cm]
    story.append(KeepTogether((prepend or []) + [
        Spacer(1, 0.5 * cm),
        Paragraph(T["top10_title"], styles["sub_title"]),
        Paragraph(T["top10_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Table(data, colWidths=col_w, style=ts),
    ]))


def _critical_profile_cards(story, util_data, T, styles, content_w):
    """Cross-utility status card for every Critical tenant."""
    # Gather all Critical tenants
    critical_set = {}
    for prefix, ud in util_data.items():
        for r in ud["rows"]:
            if r.get("status") == "Critical":
                key = (r["brand"], r["building"])
                critical_set.setdefault(key, {})
                critical_set[key][prefix] = r

    if not critical_set:
        story.append(Paragraph(T["profile_none"], styles["note"]))
        return

    prefixes = list(util_data.keys())
    lang_key = "ko" if T.get("col_bldg") == "건물" else "en"

    # Build header
    util_names = [util_data[p]["meta"].get(lang_key, p) for p in prefixes]
    hdr = ([Paragraph(T["th_tenant"], styles["table_hdr"]),
             Paragraph(T["th_bldg"],   styles["table_hdr"])] +
           [Paragraph(n, styles["table_hdr"]) for n in util_names])

    data = [hdr]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
    ])

    for ri, ((brand, bldg), util_rows) in enumerate(sorted(critical_set.items()), 1):
        row = [
            Paragraph(textwrap.shorten(str(brand), 20, placeholder="…"), styles["table_cell"]),
            Paragraph(str(bldg), styles["table_cell_c"]),
        ]
        for ci, p in enumerate(prefixes):
            r = util_rows.get(p)
            if r is None:
                cell_txt = "—"
                cell_color = None
            else:
                status = r.get("status", "Normal")
                ch = _v(r)
                pt = r.get("pct")
                cell_txt = f"{ch:+,.1f}\n{_pct_val(pt)}"
                cell_color = STATUS_COLOR_RL.get(status)
            row.append(Paragraph(cell_txt, styles["table_cell_c"]))
            if cell_color:
                col_idx = 2 + ci
                ts.add("BACKGROUND", (col_idx, ri), (col_idx, ri), cell_color)
                if cell_color == C_CRITICAL:
                    ts.add("TEXTCOLOR", (col_idx, ri), (col_idx, ri), colors.white)
        data.append(row)

    fixed_w = [4.5*cm, 1.5*cm]
    util_w  = (content_w - sum(fixed_w)) / len(prefixes)
    col_w   = fixed_w + [util_w] * len(prefixes)
    story.append(Table(data, colWidths=col_w, style=ts, repeatRows=1))


def _executive_summary(util_data, T, n_total):
    """Return a plain-language summary string for the cover page."""
    total_critical = sum(ud["status_counts"].get("Critical", 0) for ud in util_data.values())
    total_watch    = sum(ud["status_counts"].get("Watch",    0) for ud in util_data.values())
    total_alert    = sum(ud["status_counts"].get("Alert",    0) for ud in util_data.values())

    if total_critical == 0 and total_watch == 0 and total_alert == 0:
        return T["exec_all_clear"].format(n_total=n_total, n_util=len(util_data))

    parts = [T["exec_critical_lead"].format(n_critical=total_critical)]

    # Which building has the most flags?
    bldg_counts = {}
    for ud in util_data.values():
        for r in ud["rows"]:
            if r.get("status") in ("Critical", "Watch", "Alert"):
                b = r.get("building", "")
                if b:
                    bldg_counts[b] = bldg_counts.get(b, 0) + 1
    if bldg_counts:
        top_bldg, top_n = max(bldg_counts.items(), key=lambda x: x[1])
        if top_n > 1:
            parts.append(T["exec_building_concentration"].format(bldg=top_bldg, n=top_n))

    # Vacancy count across all utilities
    vacancy_brands = set()
    for ud in util_data.values():
        for r in ud["rows"]:
            cu = r.get("curr")
            pv = r.get("prev")
            if (cu is not None and not pd.isna(cu) and float(cu) < 0.01
                    and pv is not None and not pd.isna(pv)):
                vacancy_brands.add((r.get("brand"), r.get("building")))
    if vacancy_brands:
        parts.append(T["exec_vacancy"].format(n=len(vacancy_brands)))

    # New tenants
    new_brands = set()
    for ud in util_data.values():
        for r in ud["rows"]:
            if (r.get("status") == "No Data"
                    and (r.get("prev") is None or pd.isna(r.get("prev", np.nan)))
                    and r.get("curr") is not None
                    and not pd.isna(r.get("curr", np.nan))):
                new_brands.add((r.get("brand"), r.get("building")))
    if new_brands:
        parts.append(T["exec_new"].format(n=len(new_brands)))

    if total_watch > 0 or total_alert > 0:
        parts.append(T["exec_watch_alert"].format(n_watch=total_watch, n_alert=total_alert))

    return " ".join(parts)


def _action_checklist_page(story, util_data, T, styles, content_w):
    """Append a printable action checklist as the last page before back matter."""
    story.append(_section_bar(T["checklist_title"], styles, content_w))
    story.append(Spacer(1, 0.4 * cm))

    # Collect all Critical, Watch, Alert rows across utilities, deduplicated
    seen = set()
    checklist_rows = []
    status_order = {"Critical": 0, "Watch": 1, "Alert": 2}
    for prefix, ud in util_data.items():
        unit = ud["unit"]
        for r in ud["rows"]:
            status = r.get("status")
            if status not in status_order:
                continue
            key = (r.get("brand"), r.get("building"), prefix)
            if key in seen:
                continue
            seen.add(key)
            ch  = _v(r)
            pct = r.get("pct")
            pct_str = _pct(pct) if not pd.isna(pct) else "—"
            ch_str  = f"{ch:+,.1f} {unit}" if ch != 0 else "—"
            issue_key = f"cl_issue_{status.lower()}"
            issue = T.get(issue_key, "").format(change=ch_str, pct=pct_str)
            checklist_rows.append({
                "brand":    r.get("brand", ""),
                "building": r.get("building", ""),
                "floor":    r.get("floor", ""),
                "util":     ud["name"],
                "status":   status,
                "issue":    issue,
                "order":    status_order[status],
            })

    checklist_rows.sort(key=lambda x: (x["order"], x["building"], x["brand"]))

    if not checklist_rows:
        story.append(Paragraph(T["cl_none"], styles["note"]))
        return

    # col order: 상태 | 번호 | 임차인 | 건물/층 | 항목 | 이슈 | 확인 | 해결 | 조치 비고
    hdr = [
        T["cl_th_status"], T["cl_th_no"], T["cl_th_tenant"], T["cl_th_bldg"],
        T["cl_th_utility"], T["cl_th_issue"],
        T["cl_th_check"], T["cl_th_resolve"], T["cl_th_remarks"],
    ]
    col_w = [1.4*cm, 0.9*cm, 2.8*cm, 1.6*cm, 1.8*cm,
             content_w - 15.5*cm, 1.1*cm, 1.1*cm, 4.8*cm]

    data = [[Paragraph(h, styles["table_hdr"]) for h in hdr]]
    ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS",(1, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        # 번호 column (index 1) — tight padding so number stays horizontal
        ("LEFTPADDING",   (1, 0), (1, -1), 2),
        ("RIGHTPADDING",  (1, 0), (1, -1), 2),
        ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        # checkbox columns — centered, larger font for the box character
        ("ALIGN",         (6, 0), (7, -1),  "CENTER"),
        ("FONTSIZE",      (6, 1), (7, -1),  12),
        # remarks column — extra vertical room for handwriting
        ("TOPPADDING",    (8, 1), (8, -1),  12),
        ("BOTTOMPADDING", (8, 1), (8, -1),  12),
        ("BACKGROUND",    (8, 1), (8, -1),  colors.HexColor("#FAFAFA")),
    ])

    for i, row in enumerate(checklist_rows, 1):
        bldg_floor = f"{row['building']} / {row['floor']}" if row["floor"] else row["building"]
        data.append([
            Paragraph(row["status"],   styles["table_cell_c"]),
            Paragraph(str(i),          styles["table_cell_c"]),
            Paragraph(textwrap.shorten(row["brand"], width=18, placeholder="…"), styles["table_cell"]),
            Paragraph(bldg_floor,      styles["table_cell"]),
            Paragraph(row["util"],     styles["table_cell"]),
            Paragraph(row["issue"],    styles["table_cell"]),
            Paragraph("□",            styles["table_cell"]),
            Paragraph("□",            styles["table_cell"]),
            Paragraph("",             styles["table_cell"]),
        ])
        ts.add("BACKGROUND", (0, i), (0, i), STATUS_COLOR_RL.get(row["status"], C_NORMAL))
        if row["status"] == "Critical":
            ts.add("TEXTCOLOR", (0, i), (0, i), colors.white)

    story.append(KeepTogether([
        Paragraph(T["checklist_note"], styles["note"]),
        Spacer(1, 0.2 * cm),
        Table(data, colWidths=col_w, style=ts, repeatRows=1),
    ]))


def _make_numbered_canvas(T):
    """Return a Canvas subclass that writes 'Page X of Y' in the right footer."""
    _margin   = 2 * cm
    _fmt_str  = T["footer_page"]

    class NumberedCanvas(_CanvasBase):
        def __init__(self, *args, **kwargs):
            _CanvasBase.__init__(self, *args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_page_number(total)
                _CanvasBase.showPage(self)
            _CanvasBase.save(self)

        def _draw_page_number(self, total):
            page_w = self._pagesize[0]
            self.saveState()
            self.setFont("NanumGothic", 7.5)
            self.setFillColor(C_SUBTEXT)
            self.drawRightString(
                page_w - _margin, 1.2 * cm,
                _fmt_str.format(n=self._pageNumber, total=total),
            )
            self.restoreState()

    return NumberedCanvas


def _make_page_template(doc, T):
    page_w, _ = A4
    margin = 2 * cm

    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("NanumGothic", 7.5)
        canvas.setFillColor(C_SUBTEXT)
        canvas.drawString(margin, 1.2 * cm, T["footer_left"])
        # page number is drawn by _make_numbered_canvas (needs total page count)
        canvas.setStrokeColor(C_DIVIDER)
        canvas.setLineWidth(0.4)
        canvas.line(margin, 1.5 * cm, page_w - margin, 1.5 * cm)
        canvas.restoreState()

    frame = Frame(
        margin, 1.8 * cm,
        page_w - 2 * margin, A4[1] - margin - 1.8 * cm,
        id="main", leftPadding=0, rightPadding=0,
        topPadding=0, bottomPadding=0,
    )
    return PageTemplate(id="main", frames=[frame], onPage=_footer)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_report_pdf(cur_df: pd.DataFrame, present: list,
                        tail_pct: int, context: dict = None,
                        lang: str = "en") -> bytes:
    """
    Generate a business-ready PDF report.

    Parameters
    ----------
    cur_df    : aggregated dataframe (one row per brand)
    present   : list of utility prefixes in the data
    tail_pct  : tail threshold percentage (e.g. 20)
    context   : dict with keys 'date', 'buildings', 'floors'
    lang      : 'en' or 'ko'
    """
    _ensure_fonts()

    T      = _STRINGS.get(lang, _STRINGS["en"])
    ctx    = context or {}
    q_lo   = tail_pct / 100.0
    q_hi   = 1.0 - q_lo
    styles = _make_styles()

    page_w, _  = A4
    margin     = 2 * cm
    content_w  = page_w - 2 * margin

    # Status display labels for charts
    status_labels = {k: T.get(k, k) for k in
                     ["Critical", "Watch", "Alert", "Stable", "Normal", "No Data"]}

    buf = io.BytesIO()
    doc = BaseDocTemplate(
        buf, pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=margin,  bottomMargin=2 * cm,
    )
    doc.addPageTemplates([_make_page_template(doc, T)])
    story = []

    # ── Pre-compute per-utility stats ────────────────────────────────────────
    util_data = {}
    for prefix in present:
        meta = UTILITY_META.get(prefix, {"en": prefix, "ko": prefix, "unit": ""})
        change_col = f"{prefix}_change"
        pct_col    = f"{prefix}_pct"
        prev_col   = f"{prefix}_previous"
        curr_col   = f"{prefix}_current"

        s_ch = to_numeric_series(cur_df[change_col]) if change_col in cur_df.columns else pd.Series(dtype=float)
        s_pt = to_numeric_series(cur_df[pct_col])    if pct_col    in cur_df.columns else pd.Series(dtype=float)
        s_cu = to_numeric_series(cur_df[curr_col])   if curr_col   in cur_df.columns else pd.Series(dtype=float)
        s_pv = to_numeric_series(cur_df[prev_col])   if prev_col   in cur_df.columns else pd.Series(dtype=float)

        valid = s_ch.notna() & s_pt.notna()
        s_ch_v = s_ch[valid]; s_pt_v = s_pt[valid]
        if s_ch_v.empty:
            continue

        hi_c = float(s_ch_v.quantile(q_hi))
        lo_c = float(s_ch_v.quantile(q_lo))
        hi_p = float(s_pt_v.quantile(q_hi)) if not s_pt_v.empty else np.nan
        lo_p = float(s_pt_v.quantile(q_lo)) if not s_pt_v.empty else np.nan

        rows = []
        for idx in cur_df.index:
            ch = s_ch.at[idx] if idx in s_ch.index else np.nan
            pt = s_pt.at[idx] if idx in s_pt.index else np.nan
            pv = s_pv.at[idx] if idx in s_pv.index else np.nan
            cu = s_cu.at[idx] if idx in s_cu.index else np.nan
            status = _classify(ch, pt, hi_c, lo_c, hi_p, lo_p)
            rows.append({
                "brand":    str(cur_df.at[idx, "brand"])    if "brand"    in cur_df.columns else "",
                "building": str(cur_df.at[idx, "building"]) if "building" in cur_df.columns else "",
                "floor":    str(cur_df.at[idx, "floor"])    if "floor"    in cur_df.columns else "",
                "prev": pv, "curr": cu, "change": ch, "pct": pt,
                "status": status,
            })
        rows.sort(key=lambda r: (STATUS_ORDER.get(r["status"], 9),
                                 -(r["change"] if not pd.isna(r.get("change")) else 0)))

        status_counts = {}
        for r in rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

        util_data[prefix] = {
            "meta": meta, "unit": meta["unit"],
            "name": f"{meta[lang]} ({meta['unit']})" if lang == "ko"
                    else f"{meta['en']} ({meta['ko']})",
            "hi_c": hi_c, "lo_c": lo_c, "hi_p": hi_p, "lo_p": lo_p,
            "n_data":        int(valid.sum()),
            "s_cu":          s_cu.dropna(),
            "rows":          rows,
            "status_counts": status_counts,
        }

    # ═════════════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 2 * cm))
    story.append(Table(
        [[""]],
        colWidths=[content_w], rowHeights=[5],
        style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_BLUE)]),
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(T["report_title"],    styles["cover_title"]))
    story.append(Spacer(1, 0.15 * cm))
    story.append(Paragraph(T["report_subtitle"], styles["cover_sub"]))
    story.append(Spacer(1, 0.4 * cm))

    buildings      = ctx.get("buildings", "All")
    floors         = ctx.get("floors",    "All")
    rep_date       = ctx.get("date",      str(date.today()))
    billing_period = ctx.get("period",    rep_date)

    def _meta_name(prefix):
        m = UTILITY_META.get(prefix, {"en": prefix, "ko": prefix})
        return m[lang] if lang == "ko" else f"{m['en']} ({m['ko']})"

    meta_data = [
        (T["date"],            rep_date),
        (T.get("period_label", "Billing Period" if lang == "en" else "청구 기간"), billing_period),
        (T["buildings"],       buildings),
        (T["floors"],          floors),
        (T["tenants_n"],       str(len(cur_df))),
        (T["threshold_label"], T["threshold_value"].format(pct=tail_pct)),
        (T["utilities"],       ", ".join(_meta_name(p) for p in util_data)),
    ]
    story.append(Table(
        [[Paragraph(k, styles["table_cell"]), Paragraph(v, styles["table_cell"])]
         for k, v in meta_data],
        colWidths=[4 * cm, content_w - 4 * cm],
        style=TableStyle([
            ("FONTNAME",       (0, 0), (0, -1), "NanumGothic-Bold"),
            ("TEXTCOLOR",      (0, 0), (0, -1), C_NAVY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.HexColor("#F5F7FA"), colors.white]),
            ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
            ("TOPPADDING",     (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
            ("LEFTPADDING",    (0, 0), (-1, -1), 8),
        ]),
    ))
    story.append(Spacer(1, 0.8 * cm))

    # Summary overview table
    story.append(Paragraph(T["summary_title"], styles["sub_title"]))

    ov_headers = [T["col_utility"], T["col_tenants"], T["col_avg_use"],
                  T["col_avg_pct"], T["col_critical"], T["col_watch"],
                  T["col_alert"],   T["col_status"]]
    ov_data = [[Paragraph(h, styles["table_hdr"]) for h in ov_headers]]

    ov_ts = TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  C_NAVY),
        ("FONTSIZE",       (0, 0), (-1, -1), 8),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  C_WHITE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F5F7FA"), colors.white]),
        ("GRID",           (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",     (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 5),
        ("LEFTPADDING",    (0, 0), (-1, -1), 5),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
    ])

    for ri, (prefix, ud) in enumerate(util_data.items(), 1):
        sc   = ud["status_counts"]
        n_cr = sc.get("Critical", 0)
        n_wt = sc.get("Watch",    0)
        n_al = sc.get("Alert",    0)
        s_cu = ud["s_cu"]
        avg_pct = float(np.mean([r["pct"] for r in ud["rows"]
                                 if not pd.isna(r.get("pct"))]) or np.nan)

        if n_cr > 0:
            ov_status, ov_c = T["ov_critical"], C_CRITICAL
            ov_tc = colors.white
        elif n_wt + n_al > 0:
            ov_status, ov_c = T["ov_watch"],    C_ALERT
            ov_tc = C_TEXT
        else:
            ov_status, ov_c = T["ov_normal"],   C_STABLE
            ov_tc = colors.white

        ov_data.append([
            Paragraph(_meta_name(prefix),                           styles["table_cell"]),
            Paragraph(str(ud["n_data"]),                            styles["table_cell_c"]),
            Paragraph(_fmt(s_cu.mean()) + f" ({ud['unit']})" if not s_cu.empty else "—", styles["table_cell_c"]),
            Paragraph(_pct_val(avg_pct),                            styles["table_cell_c"]),
            Paragraph(str(n_cr) if n_cr > 0 else "—",              styles["table_cell_c"]),
            Paragraph(str(n_wt) if n_wt > 0 else "—",              styles["table_cell_c"]),
            Paragraph(str(n_al) if n_al > 0 else "—",              styles["table_cell_c"]),
            Paragraph(ov_status,                                    styles["table_cell_c"]),
        ])
        ov_ts.add("BACKGROUND", (7, ri), (7, ri), ov_c)
        ov_ts.add("TEXTCOLOR",  (7, ri), (7, ri), ov_tc)

    story.append(Table(
        ov_data,
        colWidths=[4.0*cm, 1.8*cm, 2.4*cm, 2.2*cm, 1.4*cm, 1.4*cm, 1.4*cm, 2.6*cm],
        style=ov_ts, repeatRows=1,
    ))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(T["cover_note"].format(pct=tail_pct), styles["note"]))
    story.append(Spacer(1, 0.8 * cm))

    # Building totals
    bldg_tbl = _building_totals_table(util_data, T, styles, content_w)
    if bldg_tbl:
        story.append(KeepTogether([
            Paragraph(T["bldg_totals_title"], styles["sub_title"]),
            bldg_tbl,
        ]))
        story.append(Spacer(1, 0.6 * cm))

    # Data coverage
    story.append(KeepTogether([
        Paragraph(T["coverage_title"], styles["sub_title"]),
        _data_coverage_table(util_data, T, styles, content_w),
    ]))
    story.append(Spacer(1, 0.6 * cm))

    # Seasonal note
    seasonal = _seasonal_note(billing_period, T)
    if seasonal:
        seasonal_style = ParagraphStyle(
            "Seasonal", parent=styles["note"],
            fontSize=9, leading=14,
            borderPad=8, borderColor=C_ALERT, borderWidth=1.2,
            backColor=colors.HexColor("#FFFBF0"),
            leftIndent=8, rightIndent=8,
        )
        story.append(Paragraph(f"<b>{T['seasonal_title']}:</b> {seasonal}", seasonal_style))
        story.append(Spacer(1, 0.6 * cm))

    # Executive summary
    exec_style = ParagraphStyle(
        "ExecSummary", parent=styles["note"],
        fontSize=10, leading=16,
        borderPad=10, borderColor=C_BLUE, borderWidth=1.5,
        backColor=colors.HexColor("#F0F4FA"),
        leftIndent=10, rightIndent=10,
        spaceBefore=4, spaceAfter=4,
    )
    story.append(KeepTogether([
        Paragraph(T["exec_title"], styles["sub_title"]),
        Paragraph(_executive_summary(util_data, T, len(cur_df)), exec_style),
    ]))

    # ── Status Legend (once, at top) ─────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    _leg_hdr = [T["th_leg_status"], T["th_leg_meaning"],
                T["th_leg_condition"], T["th_leg_action"]]
    _leg_data = [[Paragraph(h, styles["table_hdr"]) for h in _leg_hdr]]
    _leg_ts = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  C_WHITE),
        ("FONTSIZE",      (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ])
    for _ri, _key in enumerate(["Critical", "Watch", "Alert", "Normal", "Stable", "No Data"], 1):
        _leg_data.append([
            Paragraph(T.get(_key, _key),                                   styles["table_cell"]),
            Paragraph(T.get(f"desc_{_key}", ""),                           styles["table_cell"]),
            Paragraph(T.get(f"cond_{_key}", "").format(pct=tail_pct),     styles["table_cell"]),
            Paragraph(T.get(f"act_{_key}", ""),                            styles["table_cell"]),
        ])
        _leg_ts.add("BACKGROUND", (0, _ri), (0, _ri), STATUS_COLOR_RL.get(_key, C_NORMAL))
        if _key == "Critical":
            _leg_ts.add("TEXTCOLOR", (0, _ri), (0, _ri), colors.white)

    _leg_note_text = T.get("leg_condition_note", "").format(pct=tail_pct)
    story.append(KeepTogether([
        Paragraph(T["legend_title"], styles["sub_title"]),
        Table(_leg_data,
              colWidths=[2.2*cm, 3.4*cm, 5.6*cm, content_w - 11.2*cm],
              style=_leg_ts, repeatRows=1),
        Spacer(1, 0.15 * cm),
        Paragraph(_leg_note_text, styles["note"]),
    ]))

    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 2: CRITICAL ALERTS SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    _critical_alerts_page(story, util_data, T, styles, content_w, tail_pct)

    # ═════════════════════════════════════════════════════════════════════════
    # CRITICAL TENANT PROFILES
    # ═════════════════════════════════════════════════════════════════════════
    story.append(_section_bar(T["profile_title"], styles, content_w))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(T["profile_note"], styles["note"]))
    story.append(Spacer(1, 0.3 * cm))
    _critical_profile_cards(story, util_data, T, styles, content_w)
    story.append(PageBreak())

    # ═════════════════════════════════════════════════════════════════════════
    # PAGE 3: VACANCY & UNUSUAL ACTIVITY
    # ═════════════════════════════════════════════════════════════════════════
    _vacancy_section(story, util_data, T, styles, content_w)

    # ═════════════════════════════════════════════════════════════════════════
    # PER-UTILITY SECTIONS
    # ═════════════════════════════════════════════════════════════════════════
    for prefix, ud in util_data.items():
        unit = ud["unit"]
        rows = ud["rows"]
        s_cu = ud["s_cu"]
        hi_c, lo_c = ud["hi_c"], ud["lo_c"]
        hi_p, lo_p = ud["hi_p"], ud["lo_p"]
        name = ud["name"]

        # Section header
        story.append(_section_bar(
            T["section_analysis"].format(name=name), styles, content_w,
        ))
        story.append(Spacer(1, 0.4 * cm))

        # Distribution stats
        hdr = [T["col_metric"], f"{T['col_value']} ({unit})", T["col_meaning"]]
        stat_rows = [hdr]
        if not s_cu.empty:
            stat_rows += [
                (T["stat_n"],   str(ud["n_data"]),               T["stat_n_m"]),
                (T["stat_med"], _fmt(s_cu.median()),              T["stat_med_m"]),
                (T["stat_avg"], _fmt(s_cu.mean()),                T["stat_avg_m"]),
                (T["stat_std"], _fmt(s_cu.std()),                 T["stat_std_m"]),
                (T["stat_p80"], _fmt(s_cu.quantile(0.80)),        T["stat_p80_m"]),
                (T["stat_p20"], _fmt(s_cu.quantile(0.20)),        T["stat_p20_m"]),
                (T["stat_hi"].format(pct=tail_pct),
                 _fmt(hi_c, sign=True),
                 T["stat_hi_m"].format(pct=tail_pct)),
                (T["stat_lo"].format(pct=tail_pct),
                 _fmt(lo_c, sign=True),
                 T["stat_lo_m"].format(pct=tail_pct)),
            ]

        col_w = [3.8 * cm, 2.8 * cm, content_w - 6.6 * cm]
        dist_block = [
            Paragraph(T["dist_title"], styles["sub_title"]),
            Paragraph(T["dist_note"],  styles["note"]),
            _stats_table(stat_rows, col_w, styles),
        ]

        # Median vs mean interpretation
        if not s_cu.empty:
            med_v = float(s_cu.median())
            avg_v = float(s_cu.mean())
            med_str = _fmt(med_v, unit)
            avg_str = _fmt(avg_v, unit)
            ratio = avg_v / med_v if med_v != 0 else 1.0
            if ratio > 1.15:
                interp_key = "med_vs_avg_high"
            elif ratio < 0.85:
                interp_key = "med_vs_avg_low"
            else:
                interp_key = "med_vs_avg_even"
            dist_block += [
                Spacer(1, 0.25 * cm),
                Paragraph(T[interp_key].format(avg=avg_str, med=med_str), styles["note"]),
            ]

        story.append(KeepTogether(dist_block))
        story.append(Spacer(1, 0.6 * cm))

        # Charts
        hist_buf   = _chart_histogram(
            s_cu.tolist(),
            hi=float(s_cu.quantile(q_hi)) if not s_cu.empty else 0,
            lo=float(s_cu.quantile(q_lo)) if not s_cu.empty else 0,
            unit=unit,
            title=T["chart_hist_title"].format(name=name),
            xlabel_suffix="n" if lang == "en" else "임차인 수",
        )
        change_buf = _chart_change_bar(
            rows, unit=unit,
            title=T["chart_bar_title"].format(name=name),
            status_labels=status_labels,
        )
        donut_buf  = _chart_status_donut(
            ud["status_counts"],
            title=T["chart_donut_title"],
            status_labels=status_labels,
        )

        # Keep charts title with the first chart that follows it
        first_chart_items = [Paragraph(T["charts_title"], styles["sub_title"])]
        if hist_buf:
            first_chart_items += _img_flowable(
                hist_buf, width_cm=17, styles=styles,
                caption=T["chart_hist_caption"].format(name=name, pct=tail_pct),
            )
        story.append(KeepTogether(first_chart_items))

        if change_buf:
            story += _img_flowable(change_buf, 17, styles)
        if donut_buf:
            story += _img_flowable(
                donut_buf, width_cm=10, styles=styles,
                caption=T["chart_bar_caption"],
            )

        bldg_chart_buf = _chart_building_comparison(util_data, T, prefix)
        if bldg_chart_buf:
            story += _img_flowable(
                bldg_chart_buf, width_cm=12, styles=styles,
                caption=T["bldg_chart_caption"],
            )

        story.append(PageBreak())

        # ── Tenant Detail ────────────────────────────────────────────────────
        # col order: 상태 | 번호 | 임차인 | 건물 | 층 | 이전 | 이번 | 변화량 | 변화율
        _fixed_w = (2.1 + 0.9 + 1.0 + 1.0 + 2.0 + 2.0 + 2.0 + 1.4) * cm
        cw = [2.1*cm, 0.9*cm, content_w - _fixed_w, 1.0*cm, 1.0*cm, 2.0*cm, 2.0*cm, 2.0*cm, 1.4*cm]

        # Critical + Watch + Alert (sorted by STATUS_ORDER: Critical → Watch → Alert)
        alert_rows = [r for r in rows if r["status"] in ("Critical", "Watch", "Alert")]
        _sec_bar = _section_bar(T["section_detail"].format(name=name), styles, content_w)

        if alert_rows:
            # Keep section bar with the flagged sub-heading (not the full table)
            story.append(KeepTogether([
                _sec_bar,
                Spacer(1, 0.3 * cm),
                Paragraph(T["flagged_title"], styles["sub_title"]),
                Paragraph(T["flagged_note"].format(pct=tail_pct), styles["note"]),
            ]))
            story.append(_detail_table(alert_rows, unit, cw, styles, T))
            story.append(Spacer(1, 0.6 * cm))
            _top10_section(story, rows, unit, name, T, styles, content_w)
        else:
            # Keep section bar with the top-10 heading
            _top10_section(story, rows, unit, name, T, styles, content_w,
                           prepend=[_sec_bar, Spacer(1, 0.3 * cm)])


        story.append(PageBreak())

    # ── Action checklist ─────────────────────────────────────────────────────
    _action_checklist_page(story, util_data, T, styles, content_w)

    # ── Back matter ──────────────────────────────────────────────────────────
    story.append(Spacer(1, 3 * cm))
    story.append(Table(
        [[""]],
        colWidths=[content_w], rowHeights=[4],
        style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), C_BLUE)]),
    ))
    story.append(Spacer(1, 0.4 * cm))
    story.append(Paragraph(T["end_title"], styles["sub_title"]))
    story.append(Paragraph(T["end_note"].format(date=rep_date), styles["note"]))

    doc.build(story, canvasmaker=_make_numbered_canvas(T))
    buf.seek(0)
    return buf.read()
