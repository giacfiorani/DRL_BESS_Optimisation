import pandas as pd

def audit_alignment(prices_df: pd.DataFrame, carbon_df: pd.DataFrame, ts_col="timestamp"):
    """
    Cross-check timestamp coverage + missing values between two time series tables.
    Assumes both have a timestamp column (tz-naive or tz-aware but consistent).
    """

    # --- 0) Basic hygiene: unique timestamps ---
    dup_p = prices_df[ts_col].duplicated().sum()
    dup_c = carbon_df[ts_col].duplicated().sum()
    if dup_p or dup_c:
        print(f"[WARN] Duplicate timestamps: prices={dup_p}, carbon={dup_c}")

    # --- 1) Timestamp set differences ---
    p_ts = pd.Index(prices_df[ts_col])
    c_ts = pd.Index(carbon_df[ts_col])

    in_price_not_carbon = p_ts.difference(c_ts)
    in_carbon_not_price = c_ts.difference(p_ts)

    print("---- Timestamp coverage ----")
    print(f"prices rows: {len(prices_df):,} | carbon rows: {len(carbon_df):,}")
    print(f"in price, not carbon: {len(in_price_not_carbon):,}")
    print(f"in carbon, not price: {len(in_carbon_not_price):,}")

    # show small samples for debugging
    if len(in_price_not_carbon) > 0:
        print("\nSample timestamps in PRICE but not CARBON:")
        print(pd.Series(in_price_not_carbon[:10]).to_string(index=False))
    if len(in_carbon_not_price) > 0:
        print("\nSample timestamps in CARBON but not PRICE:")
        print(pd.Series(in_carbon_not_price[:10]).to_string(index=False))

    # --- 2) Row-level NaN checks on matched timestamps (outer merge) ---
    merged_outer = prices_df.merge(
        carbon_df,
        on=ts_col,
        how="outer",
        indicator=True,
        suffixes=("_price", "_carbon")
    )

    # Rows that exist only on one side (same as set differences, but tabular)
    only_price = merged_outer[merged_outer["_merge"] == "left_only"]
    only_carbon = merged_outer[merged_outer["_merge"] == "right_only"]

    print("\n---- Outer-merge row diagnostics ----")
    print(f"left_only  (price-only rows):  {len(only_price):,}")
    print(f"right_only (carbon-only rows): {len(only_carbon):,}")
    print(f"both (matched timestamps):     {len(merged_outer) - len(only_price) - len(only_carbon):,}")

    # --- 3) Missing values inside matched rows ---
    both = merged_outer[merged_outer["_merge"] == "both"].copy()

    price_cols = [c for c in prices_df.columns if c != ts_col]
    carbon_cols = [c for c in carbon_df.columns if c != ts_col]

    # Any NaNs in matched rows
    nans_price_in_both = both[price_cols].isna().any(axis=1).sum() if price_cols else 0
    nans_carbon_in_both = both[carbon_cols].isna().any(axis=1).sum() if carbon_cols else 0

    print("\n---- NaNs within matched timestamps ----")
    print(f"matched rows with any NaN in PRICE cols:  {nans_price_in_both:,}")
    print(f"matched rows with any NaN in CARBON cols: {nans_carbon_in_both:,}")

    # --- 4) Optional: check 30-min cadence gaps on each series ---
    def cadence_gaps(df: pd.DataFrame, label: str):
        ts_sorted = df[[ts_col]].dropna().drop_duplicates().sort_values(ts_col)[ts_col]
        dt = ts_sorted.diff().dropna()
        # expected 30 minutes
        bad = dt[dt != pd.Timedelta(minutes=30)]
        print(f"\n---- Cadence check ({label}) ----")
        print(f"bad intervals (not 30 min): {len(bad):,}")
        if len(bad) > 0:
            out = pd.DataFrame({
                "prev": ts_sorted.shift(1).loc[bad.index].values,
                "curr": ts_sorted.loc[bad.index].values,
                "delta": bad.values
            }).head(10)
            print("sample gaps:")
            print(out.to_string(index=False))

    cadence_gaps(prices_df, "prices")
    cadence_gaps(carbon_df, "carbon")

    return {
        "in_price_not_carbon": in_price_not_carbon,
        "in_carbon_not_price": in_carbon_not_price,
        "only_price_df": only_price,
        "only_carbon_df": only_carbon,
        "merged_outer": merged_outer
    }