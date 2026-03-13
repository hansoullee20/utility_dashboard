"""brand_normalize.py — Brand name normalizer + cross-sheet reconciliation.

Reconciles brand naming differences between sheets like:
  - 브랜드별 집계 내역 (집계)
  - 수도광열비 부과 내역 (부과)

Common discrepancies:
  - Parenthetical previous-tenant suffixes: (전.xxx), (구:xxx), (舊.xxx)
  - Spacing: "족발야시장" vs "족발 야시장"
  - Date annotations: "(11/30해지)", "(7/31)"
  - Character variants: "에이" vs "A"
"""
from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd


# ── Synonym persistence ──────────────────────────────────────────────────────

_SYNONYM_FILE = Path(__file__).parent / "brand_synonyms.json"


def load_synonyms() -> dict[str, str]:
    """Load saved brand synonym mappings {alt_norm: canonical_norm}."""
    if _SYNONYM_FILE.exists():
        try:
            return json.loads(_SYNONYM_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_synonyms(synonyms: dict[str, str]) -> None:
    """Persist brand synonym mappings to JSON."""
    _SYNONYM_FILE.write_text(
        json.dumps(synonyms, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── Normalizer ────────────────────────────────────────────────────────────────

def normalize_brand(name: str) -> str:
    """Normalize brand name for fuzzy cross-sheet matching.

    Strips previous-tenant markers, date annotations, whitespace,
    and common character variants.
    """
    s = str(name).strip()
    if s in ("nan", "", "NaN"):
        return ""
    # 1. Remove trailing date annotations (with or without parens)
    s = re.sub(r"\s*\(?\d{1,2}/\d{1,2}\s*해지?\)?\s*$", "", s)
    # 2. Truncate at previous-tenant markers: (전, (구, (舊 — anywhere in string
    s = re.sub(r"\s*\((?:전|구|舊)[^)]*\).*$", "", s)
    s = re.sub(r"\s*\((?:전|구|舊)[^)]*$", "", s)  # unclosed paren
    # 3. Remove ALL trailing parentheticals repeatedly
    while True:
        s2 = re.sub(r"\s*\([^)]*\)?\s*$", "", s)
        if s2 == s:
            break
        s = s2
    # 4. Remove ALL spaces for fuzzy comparison
    s = re.sub(r"\s+", "", s)
    # 5. Common character normalization
    s = s.replace("에이", "A")
    # 6. Double consonant / vowel variants
    s = s.replace("죠", "조").replace("쬬", "조")
    return s


# ── Cross-sheet reconciliation ────────────────────────────────────────────────

def _extract_brands(
    raw: pd.DataFrame,
    brand_col: int,
    bldg_col: int = 2,
    unit_col: int | None = None,
    start_row: int = 5,
    synonyms: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Extract (brand_raw, brand_norm, building, unit) from raw Excel sheet."""
    df = raw.iloc[start_row:].copy()
    df = df[df[bldg_col].astype(str).str.strip().isin({"A", "B", "C", "D"})].copy()
    out = pd.DataFrame()
    out["orig_row"] = df.index.values  # preserve original row in raw DF
    out["brand_raw"] = df[brand_col].astype(str).str.strip().values
    out["brand_norm"] = out["brand_raw"].apply(normalize_brand)
    if synonyms:
        out["brand_norm"] = out["brand_norm"].replace(synonyms)
    out["building"] = df[bldg_col].astype(str).str.strip().values
    if unit_col is not None and unit_col in df.columns:
        out["unit"] = df[unit_col].astype(str).str.strip().values
    else:
        out["unit"] = ""
    out = out[out["brand_norm"] != ""].reset_index(drop=True)
    return out


def reconcile_sheets(
    sheet_a_raw: pd.DataFrame,
    sheet_b_raw: pd.DataFrame,
    a_brand_col: int = 10,
    b_brand_col: int = 9,
    a_unit_col: int = 4,
    b_unit_col: int = 4,
    a_totals: dict[str, int] | None = None,
    b_totals: dict[str, int] | None = None,
    synonyms: dict[str, str] | None = None,
) -> dict:
    """Reconcile two sheets and return match/mismatch report.

    Parameters
    ----------
    sheet_a_raw, sheet_b_raw : raw DataFrames (header=None) from pd.read_excel
    a_brand_col, b_brand_col : column index for brand name
    a_unit_col, b_unit_col   : column index for unit/room number
    a_totals, b_totals       : {label: col_index} for numeric columns to cross-check
                                e.g. {"전용": 13, "공용": 14, "합계": 15}
    synonyms                 : {alt_norm: canonical_norm} user-confirmed mappings

    Returns
    -------
    dict with keys: matched, fuzzy_matched, only_a, only_b, amount_mismatches
    """
    a_df = _extract_brands(sheet_a_raw, a_brand_col, unit_col=a_unit_col, synonyms=synonyms)
    b_df = _extract_brands(sheet_b_raw, b_brand_col, unit_col=b_unit_col, synonyms=synonyms)

    # Build lookup: (norm, building, unit) → row index + raw name
    def _build_lookup(df):
        by_nbu = {}  # (norm, bldg, unit) → df index
        by_nb = {}   # (norm, bldg) → df index — fallback
        for i, r in df.iterrows():
            key3 = (r["brand_norm"], r["building"], r["unit"])
            key2 = (r["brand_norm"], r["building"])
            by_nbu[key3] = i
            by_nb.setdefault(key2, i)
        return by_nbu, by_nb

    a_nbu, a_nb = _build_lookup(a_df)
    b_nbu, b_nb = _build_lookup(b_df)

    matched = []       # exact raw match
    fuzzy_matched = [] # matched after normalization (different raw names)
    only_a = []
    only_b_keys = set(b_nbu.keys())

    for key3, ai in a_nbu.items():
        a_raw = a_df.at[ai, "brand_raw"]
        a_orig = int(a_df.at[ai, "orig_row"])
        # Try exact (norm, bldg, unit)
        if key3 in b_nbu:
            bi = b_nbu[key3]
            b_raw = b_df.at[bi, "brand_raw"]
            b_orig = int(b_df.at[bi, "orig_row"])
            entry = {
                "brand_a": a_raw, "brand_b": b_raw,
                "building": key3[1], "unit": key3[2],
                "a_orig_row": a_orig, "b_orig_row": b_orig,
            }
            if a_raw == b_raw:
                matched.append(entry)
            else:
                fuzzy_matched.append(entry)
            b_nbu.pop(key3, None)
            continue
        # Fallback: (norm, bldg)
        key2 = (key3[0], key3[1])
        if key2 in b_nb:
            bi = b_nb[key2]
            b_raw = b_df.at[bi, "brand_raw"]
            b_orig = int(b_df.at[bi, "orig_row"])
            entry = {
                "brand_a": a_raw, "brand_b": b_raw,
                "building": key2[1], "unit": key3[2],
                "a_orig_row": a_orig, "b_orig_row": b_orig,
            }
            if a_raw == b_raw:
                matched.append(entry)
            else:
                fuzzy_matched.append(entry)
            # Remove from b lookups
            for k3 in list(b_nbu.keys()):
                if b_nbu[k3] == bi:
                    b_nbu.pop(k3)
                    break
            continue
        only_a.append({
            "brand": a_raw, "building": key3[1], "unit": key3[2],
            "norm": key3[0], "orig_row": a_orig,
        })

    only_b_list = [
        {"brand": b_df.at[bi, "brand_raw"],
         "building": b_df.at[bi, "building"],
         "unit": b_df.at[bi, "unit"],
         "norm": b_df.at[bi, "brand_norm"],
         "orig_row": int(b_df.at[bi, "orig_row"])}
        for bi in b_nbu.values()
    ]

    # ── Fuzzy matching pass for remaining unmatched ──────────────────────
    # Match by (building, unit) first, then by name similarity
    still_a, still_b = [], list(only_b_list)
    fuzzy_suggested = []
    b_by_bldg_unit = {}
    for item in still_b:
        b_by_bldg_unit.setdefault((item["building"], item["unit"]), []).append(item)

    for a_item in only_a:
        key_bu = (a_item["building"], a_item["unit"])
        candidates = b_by_bldg_unit.get(key_bu, [])
        best, best_ratio = None, 0.0
        for b_item in candidates:
            ratio = SequenceMatcher(None, a_item["norm"], b_item["norm"]).ratio()
            if ratio > best_ratio:
                best, best_ratio = b_item, ratio

        # Also try same-building candidates if no unit match
        if best is None or best_ratio < 0.5:
            for b_item in still_b:
                if b_item["building"] != a_item["building"]:
                    continue
                ratio = SequenceMatcher(None, a_item["norm"], b_item["norm"]).ratio()
                if ratio > best_ratio:
                    best, best_ratio = b_item, ratio

        if best is not None and best_ratio >= 0.5:
            fuzzy_suggested.append({
                "brand_a": a_item["brand"], "brand_b": best["brand"],
                "norm_a": a_item["norm"], "norm_b": best["norm"],
                "building": a_item["building"],
                "unit_a": a_item["unit"], "unit_b": best["unit"],
                "similarity": round(best_ratio * 100),
                "a_orig_row": a_item["orig_row"],
                "b_orig_row": best["orig_row"],
            })
            still_b.remove(best)
            # rebuild lookup
            b_by_bldg_unit = {}
            for item in still_b:
                b_by_bldg_unit.setdefault((item["building"], item["unit"]), []).append(item)
        else:
            still_a.append(a_item)

    only_a = [{k: v for k, v in x.items() if k not in ("norm", "orig_row")} for x in still_a]
    only_b = [{k: v for k, v in x.items() if k not in ("norm", "orig_row")} for x in still_b]

    # ── Amount cross-check for matched pairs ──────────────────────────────
    amount_mismatches = []
    if a_totals and b_totals:
        for entry in matched + fuzzy_matched:
            a_orig = entry["a_orig_row"]
            b_orig = entry["b_orig_row"]
            a_row = sheet_a_raw.iloc[a_orig] if a_orig < len(sheet_a_raw) else None
            b_row = sheet_b_raw.iloc[b_orig] if b_orig < len(sheet_b_raw) else None
            if a_row is None or b_row is None:
                continue
            for label in a_totals:
                if label not in b_totals:
                    continue
                a_val = pd.to_numeric(a_row.iloc[a_totals[label]], errors="coerce")
                b_val = pd.to_numeric(b_row.iloc[b_totals[label]], errors="coerce")
                if pd.isna(a_val) or pd.isna(b_val):
                    continue
                if abs(a_val - b_val) > 1:
                    amount_mismatches.append({
                        "brand_a": entry["brand_a"],
                        "brand_b": entry["brand_b"],
                        "building": entry["building"],
                        "field": label,
                        "a_value": a_val,
                        "b_value": b_val,
                        "diff": a_val - b_val,
                    })

    return {
        "matched": matched,
        "fuzzy_matched": fuzzy_matched,
        "fuzzy_suggested": fuzzy_suggested,
        "only_a": only_a,
        "only_b": only_b,
        "amount_mismatches": amount_mismatches,
        "summary": {
            "a_total": len(a_df),
            "b_total": len(b_df),
            "exact_match": len(matched),
            "fuzzy_match": len(fuzzy_matched),
            "fuzzy_suggested": len(fuzzy_suggested),
            "only_a": len(only_a),
            "only_b": len(only_b),
            "amount_mismatches": len(amount_mismatches),
        },
    }


# ── Multi-sheet brand name inconsistency finder ──────────────────────────────

# Sheet configs: sheet_name → brand column index
_SHEET_BRAND_COLS: dict[str, int] = {
    "브랜드별 집계 내역":    10,
    "수도광열비 부과 내역":   9,
    "수도 사용 내역":        9,
    "온수 사용 내역":        9,
    "전체 전기 사용내역":     9,
}

_SHEET_SHORT: dict[str, str] = {
    "브랜드별 집계 내역":    "브랜드별 집계 내역",
    "수도광열비 부과 내역":   "수도광열비 부과 내역",
    "수도 사용 내역":        "수도 사용 내역",
    "온수 사용 내역":        "온수 사용 내역",
    "전체 전기 사용내역":     "전체 전기 사용내역",
}


def find_name_inconsistencies(
    file_data: bytes,
    sheet_names: list[str],
    synonyms: dict[str, str] | None = None,
) -> list[dict]:
    """Find brands that have different raw names across sheets.

    Returns list of dicts: {normalized, variants: {sheet_short: raw_name}}
    Only returns entries where at least 2 different raw names exist.
    """
    import io

    # Match sheet names (handle trailing spaces)
    available = {}
    for sn in sheet_names:
        stripped = sn.strip()
        for cfg_name, brand_col in _SHEET_BRAND_COLS.items():
            if stripped == cfg_name.strip():
                available[sn] = (cfg_name, brand_col)
                break

    # Extract brands per sheet
    norm_to_raw: dict[str, dict[str, str]] = {}
    for actual_name, (cfg_name, brand_col) in available.items():
        raw = pd.read_excel(
            io.BytesIO(file_data), sheet_name=actual_name,
            header=None, engine="calamine",
        )
        df = raw.iloc[5:].copy()
        df = df[df[2].astype(str).str.strip().isin({"A", "B", "C", "D"})].copy()
        brands = df[brand_col].astype(str).str.strip()
        brands = brands[~brands.isin({"nan", "", "NaN"})].unique()

        short = _SHEET_SHORT.get(cfg_name, cfg_name[:4])
        for brand_raw in brands:
            norm = normalize_brand(brand_raw)
            if not norm:
                continue
            if synonyms and norm in synonyms:
                norm = synonyms[norm]
            norm_to_raw.setdefault(norm, {})
            if short not in norm_to_raw[norm]:
                norm_to_raw[norm][short] = brand_raw

    # Filter to inconsistent entries
    result = []
    for norm, by_sheet in sorted(norm_to_raw.items()):
        raw_names = set(by_sheet.values())
        if len(raw_names) > 1:
            result.append({"normalized": norm, "variants": by_sheet})

    return result
