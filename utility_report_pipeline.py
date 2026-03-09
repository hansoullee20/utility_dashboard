"""utility_report_pipeline.py

Clean, structured pipeline for the "검침 내역" sheet in utility_report.xlsm.

What this does (mirrors the intent of Untitled.ipynb but in reusable functions):
- Read Excel with 3-row header into a MultiIndex columns DataFrame.
- Drop an entire top-level group (e.g., "구분").
- Within a target group (e.g., "동별 건물 면적 현황"), keep only chosen sub-columns.
- Keep only blocks before the "전기 배율" section.
- Normalize header strings (remove spaces/newlines) and translate key level-1 labels.
- Flatten into a tidy, English-named table.
- Trim trailing subtotal/summary rows.
- Add change/%/per-area metrics for water/hot-water/electric/heat.

Usage:
  python utility_report_pipeline.py \
    --file utility_report.xlsm \
    --sheet "검침 내역" \
    --out cleaned_utility.csv

Notes:
- This script is conservative about assumptions and fails with helpful errors.
- Designed for Pandas + openpyxl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Configuration
# ----------------------------


@dataclass(frozen=True)
class PipelineConfig:
    excel_file: Path = Path("utility_report.xlsm")
    sheet_name: str = "검침 내역"
    header_rows: tuple[int, int, int] = (2, 3, 4)

    drop_top_level_groups: tuple[str, ...] = ("구분",)

    # Within the "동별 건물 면적 현황" block, keep only these level-1 fields
    area_block_lvl0: str = "동별 건물 면적 현황"
    area_block_keep_lvl1: tuple[str, ...] = ("건물", "층수", "전용면적")

    # Keep all column blocks before this level-0 header (inclusive? here: exclusive)
    stop_before_lvl0: str = "전기 배율"

    # Rows below this often look like totals / summaries in the sample notebook
    # We trim using a rule instead of hard-coding the index.
    # If you *want* to hard-code, set trim_after_first_all_brand_zero=True.
    trim_after_first_all_brand_zero: bool = True


# ----------------------------
# Helpers
# ----------------------------


def _normalize_header_token(x: object) -> str:
    """Normalize header token: stringify, strip spaces, remove newlines."""
    return str(x).replace(" ", "").replace("\n", "").strip()


def read_utility_sheet(
    excel_file: Path | str,
    sheet_name: str,
    header_rows: Iterable[int] = (2, 3, 4),
) -> pd.DataFrame:
    """Read the Excel sheet with a 3-row header into MultiIndex columns."""
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        engine="openpyxl",
        header=list(header_rows),
    )
    return df


def drop_top_level_groups(df: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """Drop whole level-0 groups by name (works on MultiIndex columns)."""
    if not isinstance(df.columns, pd.MultiIndex):
        # If not MultiIndex, fall back to normal column drop
        return df.drop(columns=[g for g in groups if g in df.columns], errors="ignore")

    lvl0 = df.columns.get_level_values(0)
    drop_mask = lvl0.isin(list(groups))
    return df.loc[:, ~drop_mask]


def keep_only_subcolumns(
    df: pd.DataFrame,
    lvl0_group: str,
    keep_lvl1: Iterable[str],
) -> pd.DataFrame:
    """Within a given level-0 group, keep only specified level-1 headers."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    keep_lvl1 = set(keep_lvl1)

    cols = df.columns
    lvl0 = cols.get_level_values(0)
    lvl1 = cols.get_level_values(1)

    in_group = lvl0 == lvl0_group
    keep_in_group = in_group & lvl1.isin(list(keep_lvl1))

    # Keep columns that are not in the group, plus the selected ones inside the group
    keep_mask = (~in_group) | keep_in_group
    out = df.loc[:, keep_mask]
    return out


def keep_blocks_before(df: pd.DataFrame, needle_lvl0: str) -> pd.DataFrame:
    """Keep all level-0 column blocks before the first occurrence of needle.

    Fixes the notebook bug pattern by defining all_lvl0_cols unconditionally and
    handling missing needle safely.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    all_lvl0_cols = df.columns.get_level_values(0).unique()

    # Normalize for matching because the file has variants like "전기 \n배율"
    norm = pd.Index([_normalize_header_token(x) for x in all_lvl0_cols])
    needle_norm = _normalize_header_token(needle_lvl0)

    if needle_norm not in set(norm):
        # Nothing to cut
        return df

    pos = int(np.where(norm == needle_norm)[0][0])
    keep_lvl0 = all_lvl0_cols[:pos]

    keep_mask = df.columns.get_level_values(0).isin(keep_lvl0)
    return df.loc[:, keep_mask]


def normalize_multiindex_level0(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize level-0 labels (remove spaces/newlines) without disturbing other levels."""
    if not isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [_normalize_header_token(c) for c in df.columns]
        return df

    tuples = [(_normalize_header_token(t[0]),) + t[1:] for t in df.columns.to_list()]
    out = df.copy()
    out.columns = pd.MultiIndex.from_tuples(tuples)
    return out


def translate_level1_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Translate a few known Korean level-1 labels into shorter group names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df

    rename_map = {
        "급수 지침": "수도",
        "온수(급탕) 지침": "온수",
        "전기 지침": "전기",
        "FCU (냉,난방 지침)": "열요금",
    }
    return df.rename(columns=rename_map, level=1)


def flatten_after_droplevel0(df: pd.DataFrame) -> pd.DataFrame:
    """Drop top level then flatten remaining MultiIndex into clean strings.

    In the notebook, after droplevel(0), each column is a tuple like (lvl1, lvl2).
    We keep only lvl1 and normalize it.
    """
    if not isinstance(df.columns, pd.MultiIndex):
        out = df.copy()
        out.columns = [_normalize_header_token(c) for c in out.columns]
        return out

    out = df.copy()
    out2 = out.droplevel(0, axis=1)

    cleaned = []
    for col in out2.columns:
        if isinstance(col, tuple) and len(col) > 0:
            cleaned.append(_normalize_header_token(col[0]))
        else:
            cleaned.append(_normalize_header_token(col))

    out2.columns = cleaned
    return out2


def rename_to_english_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the stable English schema used in the notebook."""
    # Expected order after the notebook's flattening
    expected = [
        "건물",
        "층수",
        "전용면적",
        "전용면적",  # (평)
        "브랜드",
        "수도",
        "수도",
        "수도",
        "온수",
        "온수",
        "온수",
        "전기",
        "전기",
        "전기",
        "열요금",
        "열요금",
        "열요금",
    ]

    english = [
        "building",
        "floor",
        "size_m2",
        "size_py",
        "brand",
        "water_previous",
        "water_current",
        "water_usage_m3",
        "hwater_previous",
        "hwater_current",
        "hwater_usage_m3",
        "elect_previous",
        "elect_current",
        "elect_usage_kw",
        "heat_previous",
        "heat_current",
        "heat_usage_m3_mwh",
    ]

    out = df.copy()

    # If columns already match english, skip
    if list(out.columns) == english:
        return out

    if len(out.columns) != len(english):
        raise ValueError(
            f"Unexpected column count after flattening: got {len(out.columns)} cols, "
            f"expected {len(english)}. Columns={list(out.columns)}"
        )

    out.columns = english
    return out


def _to_numeric_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def trim_trailing_summary_rows(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Trim trailing rows that look like summaries.

    The notebook used a hard cut at index 209 based on observed structure.
    Here we implement a rule:
    - Find the first row where building and floor are NaN and brand is 0/NaN,
      and keep only rows before that.
    """
    if not config.trim_after_first_all_brand_zero:
        return df

    out = df.copy()

    # Coerce brand to numeric-ish where possible, but brand may be str; we treat "0" as zero.
    brand = out.get("brand")
    if brand is None:
        return out

    brand_is_zero = brand.isna() | (brand.astype(str).str.strip() == "0")
    building_is_na = out.get("building").isna() if "building" in out.columns else pd.Series(False, index=out.index)
    floor_is_na = out.get("floor").isna() if "floor" in out.columns else pd.Series(False, index=out.index)

    marker = building_is_na & floor_is_na & brand_is_zero

    if not marker.any():
        return out

    cut_idx = marker.idxmax()  # first True index

    # Keep strictly before cut index
    try:
        loc = out.index.get_loc(cut_idx)
        return out.iloc[:loc]
    except Exception:
        # fallback: no trim
        return out


def add_usage_metrics(
    df: pd.DataFrame,
    prefix: str,
    prev_col: str,
    curr_col: str,
    size_m2_col: str = "size_m2",
    size_py_col: str = "size_py",
) -> pd.DataFrame:
    """Add change, pct, and per-area metrics for a given utility."""
    out = df.copy()

    for c in [prev_col, curr_col, size_m2_col, size_py_col]:
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")

    _to_numeric_inplace(out, [prev_col, curr_col, size_m2_col, size_py_col])

    change = out[curr_col] - out[prev_col]

    out[f"{prefix}_change"] = change
    out[f"{prefix}_pct"] = (change / out[prev_col]) * 100.0

    out[f"{prefix}_change_per_m2"] = change / out[size_m2_col]
    out[f"{prefix}_change_per_py"] = change / out[size_py_col]

    out[f"{prefix}_curr_per_py"] = out[curr_col] / out[size_py_col]
    out[f"{prefix}_prev_per_py"] = out[prev_col] / out[size_py_col]

    out[f"{prefix}_curr_per_m2"] = out[curr_col] / out[size_m2_col]
    out[f"{prefix}_prev_per_m2"] = out[prev_col] / out[size_m2_col]

    return out


def run_pipeline(config: PipelineConfig) -> pd.DataFrame:
    df = read_utility_sheet(config.excel_file, config.sheet_name, config.header_rows)

    # 1) Drop whole groups (e.g., "구분")
    df = drop_top_level_groups(df, config.drop_top_level_groups)

    # 2) Keep only relevant subcolumns in the area block
    df = keep_only_subcolumns(df, config.area_block_lvl0, config.area_block_keep_lvl1)

    # 3) Keep only blocks before "전기 배율" (normalize-aware)
    df = keep_blocks_before(df, config.stop_before_lvl0)

    # 4) Normalize headers + translate level-1 names
    df = normalize_multiindex_level0(df)
    df = translate_level1_labels(df)

    # 5) Flatten + rename to English schema
    df_flat = flatten_after_droplevel0(df)
    df_flat = rename_to_english_schema(df_flat)

    # 6) Trim trailing summary rows
    df_flat = trim_trailing_summary_rows(df_flat, config)

    # 7) Add metrics (matches the notebook’s intent)
    df_metrics = df_flat
    df_metrics = add_usage_metrics(df_metrics, "water", "water_previous", "water_current")
    df_metrics = add_usage_metrics(df_metrics, "hwater", "hwater_previous", "hwater_current")
    df_metrics = add_usage_metrics(df_metrics, "elect", "elect_previous", "elect_current")
    df_metrics = add_usage_metrics(df_metrics, "heat", "heat_previous", "heat_current")

    return df_metrics


# ----------------------------
# CLI
# ----------------------------


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Clean utility_report.xlsm (검침 내역) into tidy table")
    p.add_argument("--file", default="utility_report.xlsm", help="Path to utility_report.xlsm")
    p.add_argument("--sheet", default="검침 내역", help="Excel sheet name")
    p.add_argument("--out", default=None, help="Output CSV path (optional)")
    args = p.parse_args(argv)

    cfg = PipelineConfig(excel_file=Path(args.file), sheet_name=args.sheet)
    df = run_pipeline(cfg)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote: {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    else:
        print(df.head())
        print(df.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
