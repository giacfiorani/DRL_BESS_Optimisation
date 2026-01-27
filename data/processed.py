import pandas as pd
from data_loader import fetch_48sp_curve, fetch_carbon_sql, fetch_market_index_range_by_settlement_date
from data_audit import audit_alignment
from eikon_rics_lists import DA_HH_RICS, SSP_RICS, SBP_RICS

start = "2021-01-01"
end   = "2023-01-01"

carbon_df = fetch_carbon_sql(start, end)
da_df  = fetch_48sp_curve(start, end, DA_HH_RICS, curve_name="DA")
ssp_df = fetch_48sp_curve(start, end, SSP_RICS,   curve_name="SSP")
sbp_df = fetch_48sp_curve(start, end, SBP_RICS,   curve_name="SBP")
mid_df = fetch_market_index_range_by_settlement_date(
    start_date=start,
    end_date=end,
    data_providers=["APXMIDP"],
)

# ---------- 1) Clean timestamps ----------
def clean_ts(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if ts_col not in df.columns:
        raise KeyError(f"Expected column '{ts_col}' not found. Columns are: {list(df.columns)}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True).dt.tz_convert(None)
    return df.sort_values(ts_col).reset_index(drop=True)

da_df     = clean_ts(da_df)
ssp_df    = clean_ts(ssp_df)
sbp_df    = clean_ts(sbp_df)
mid_df    = clean_ts(mid_df)
carbon_df = clean_ts(carbon_df)

# ---------- 2) Align common time window ----------
start_common = max(
    carbon_df["timestamp"].min(),
    da_df["timestamp"].min(),
    ssp_df["timestamp"].min(),
    sbp_df["timestamp"].min(),
    mid_df["timestamp"].min(),
)
end_common = min(
    carbon_df["timestamp"].max(),
    da_df["timestamp"].max(),
    ssp_df["timestamp"].max(),
    sbp_df["timestamp"].max(),
    mid_df["timestamp"].max(),
)

def clip(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[m].reset_index(drop=True)

da_df     = clip(da_df, start_common, end_common)
ssp_df    = clip(ssp_df, start_common, end_common)
sbp_df    = clip(sbp_df, start_common, end_common)
mid_df    = clip(mid_df, start_common, end_common)
carbon_df = clip(carbon_df, start_common, end_common)

# ---------- 3) Rename to avoid collisions ----------
da_df  = da_df.rename(columns={"price_gbp_mwh": "da_price_gbp_mwh"})
ssp_df = ssp_df.rename(columns={"price_gbp_mwh": "ssp_gbp_mwh"})
sbp_df = sbp_df.rename(columns={"price_gbp_mwh": "sbp_gbp_mwh"})

# MID uses marketIndexPrice
if "marketIndexPrice" not in mid_df.columns:
    raise KeyError(f"MID expected 'marketIndexPrice' but got {mid_df.columns.tolist()}")
mid_df = mid_df.rename(columns={"marketIndexPrice": "mid_price_gbp_mwh"})

# keep only merge columns
carbon_df = carbon_df[["timestamp", "carbon_gco2_kwh"]]
da_df     = da_df[["timestamp", "da_price_gbp_mwh", "trade_ts", "trade_date", "delivery_ts", "delivery_date", "settlement_period"]]
ssp_df    = ssp_df[["timestamp", "ssp_gbp_mwh"]]
sbp_df    = sbp_df[["timestamp", "sbp_gbp_mwh"]]
mid_df    = mid_df[["timestamp", "mid_price_gbp_mwh", "settlementDate", "settlementPeriod", "dataProvider"]]

# ---------- 4) Merge (timestamp must be DELIVERY half-hour across all) ----------
merged_df = (
    da_df
    .merge(mid_df[["timestamp", "mid_price_gbp_mwh"]], on="timestamp", how="inner")
    .merge(ssp_df, on="timestamp", how="inner")
    .merge(sbp_df, on="timestamp", how="inner")
    .merge(carbon_df, on="timestamp", how="inner")
).sort_values("timestamp").reset_index(drop=True)

# ---------- 5) Canonical time columns (delivery_ts == timestamp) ----------
merged_df["delivery_ts"] = merged_df["timestamp"]
merged_df["delivery_date"] = merged_df["delivery_ts"].dt.floor("D")

merged_df["tau"] = (
    merged_df["delivery_ts"].dt.hour * 2
    + (merged_df["delivery_ts"].dt.minute // 30)
    + 1
).astype(int)

# agent “now” time for planning = D-1
merged_df["trade_ts"] = merged_df["delivery_ts"] - pd.Timedelta(days=1)
merged_df["trade_date"] = merged_df["trade_ts"].dt.floor("D")

# ---------- 6) Save ----------
merged_df = merged_df.drop(columns=["timestamp"])

cols = [
    "trade_ts", "trade_date",
    "delivery_ts", "delivery_date",
    "tau",
    "da_price_gbp_mwh",
    "mid_price_gbp_mwh",
    "ssp_gbp_mwh",
    "sbp_gbp_mwh",
    "carbon_gco2_kwh",
]
merged_df = merged_df[cols].sort_values(["delivery_ts"]).reset_index(drop=True)

merged_df.to_parquet("data/training_data.parquet", index=False)
print("Saved:", len(merged_df), "rows")
print("Range delivery:", merged_df["delivery_ts"].min(), "→", merged_df["delivery_ts"].max())