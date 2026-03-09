"""tab_reconciliation.py — Billing ↔ Meter reconciliation expander."""
import pandas as pd
import streamlit as st

from data import to_numeric_series, read_billing_sheet, BILLING_SHEET_NAME
from lang import t


def render_reconciliation(
    file_name: str,
    file_map: dict,
    cur_df: pd.DataFrame,
    present: list[str],
    sheet_names: list[str],
) -> None:
    """Render the Billing ↔ Meter Reconciliation expander if the billing sheet is present."""
    if BILLING_SHEET_NAME not in sheet_names:
        return

    with st.expander(t("recon_expander"), expanded=False):
        st.caption(t("recon_caption"))
        try:
            bill_df = read_billing_sheet(file_name, file_map[file_name], BILLING_SHEET_NAME)

            def _keyset(df: pd.DataFrame) -> set:
                return set(zip(
                    df["brand"].astype(str).str.strip(),
                    df["building"].astype(str).str.strip(),
                ))

            bill_set  = _keyset(bill_df.dropna(subset=["brand"]).drop_duplicates(subset=["brand", "building"]))
            meter_set = _keyset(cur_df.drop_duplicates(subset=["brand", "building"]))

            billed_not_metered = sorted(bill_set  - meter_set)
            metered_not_billed = sorted(meter_set - bill_set)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"**{t('recon_billed_no_meter')}** — {len(billed_not_metered)}")
                if billed_not_metered:
                    st.dataframe(
                        pd.DataFrame(billed_not_metered, columns=["Brand", "Building"]),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.success(t("recon_all_billed"))
            with rc2:
                st.markdown(f"**{t('recon_metered_no_bill')}** — {len(metered_not_billed)}")
                if metered_not_billed:
                    st.dataframe(
                        pd.DataFrame(metered_not_billed, columns=["Brand", "Building"]),
                        hide_index=True, use_container_width=True,
                    )
                else:
                    st.success(t("recon_all_metered"))

            # Billed non-zero but zero meter usage
            shared = bill_set & meter_set
            zero_billed = []
            for br, bl in sorted(shared):
                brow = bill_df[
                    (bill_df["brand"].astype(str).str.strip() == br) &
                    (bill_df["building"].astype(str).str.strip() == bl)
                ]
                mrow = cur_df[
                    (cur_df["brand"].astype(str).str.strip() == br) &
                    (cur_df["building"].astype(str).str.strip() == bl)
                ]
                if brow.empty or mrow.empty:
                    continue
                total = to_numeric_series(brow["total"].iloc[[0]]).iloc[0] if "total" in brow.columns else float("nan")
                has_usage = any(
                    not pd.isna(to_numeric_series(mrow[f"{px}_current"]).iloc[0])
                    and to_numeric_series(mrow[f"{px}_current"]).iloc[0] > 0
                    for px in present
                    if f"{px}_current" in mrow.columns
                )
                if not pd.isna(total) and total > 0 and not has_usage:
                    zero_billed.append({"Brand": br, "Building": bl, "Billed Total (₩)": f"{int(total):,}"})

            if zero_billed:
                st.markdown(f"**{t('recon_billed_zero')}** — {len(zero_billed)}")
                st.dataframe(pd.DataFrame(zero_billed), hide_index=True, use_container_width=True)

        except Exception as e:
            st.warning(f"{t('recon_fail')}: {e}")
