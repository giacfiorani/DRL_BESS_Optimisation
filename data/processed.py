import pandas as pd
from data_loader import fetch_system_prices, fetch_carbon_sql
from data_audit import audit_alignment

start = "2021-01-01"
end   = "2023-01-01"

prices_df = fetch_system_prices(start, end)
carbon_df = fetch_carbon_sql(start, end)

# ---------- 1) Clean timestamps ----------
def clean_ts(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    df = df.copy()
    if ts_col not in df.columns:
        raise KeyError(f"Expected column '{ts_col}' not found. Columns are: {list(df.columns)}")

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df[ts_col] = df[ts_col].dt.tz_convert(None)
    df = df.sort_values(ts_col).drop_duplicates(ts_col).reset_index(drop=True)
    return df

prices_df = clean_ts(prices_df)
carbon_df = clean_ts(carbon_df)

# ---------- 2) Align common time window ----------
start_common = max(prices_df["timestamp"].min(), carbon_df["timestamp"].min())
end_common   = min(prices_df["timestamp"].max(), carbon_df["timestamp"].max())

def clip(df: pd.DataFrame, start, end) -> pd.DataFrame:
    m = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    return df.loc[m].reset_index(drop=True)

prices_df = clip(prices_df, start_common, end_common)
carbon_df = clip(carbon_df, start_common, end_common)

# ---------- 3) Merge on delivery timestamp ----------
merged_df = prices_df.merge(
    carbon_df[["timestamp", "carbon_gco2_kwh"]],
    on="timestamp",
    how="inner",
)

# ---------- 4) Add explicit time columns ----------
merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

# timestamp is TRADE time (now)
merged_df["trade_ts"] = merged_df["timestamp"]
merged_df["trade_date"] = merged_df["trade_ts"].dt.floor("D")

# delivery is tomorrow (same half-hour slot)
merged_df["delivery_ts"] = merged_df["trade_ts"] + pd.Timedelta(days=1)
merged_df["delivery_date"] = merged_df["delivery_ts"].dt.floor("D")

# tau is defined on DELIVERY day
merged_df["tau"] = (
    merged_df["delivery_ts"].dt.hour * 2
    + (merged_df["delivery_ts"].dt.minute // 30)
    + 1
)



# ---------- 5) Optional audit ----------
RUN_AUDIT = False
if RUN_AUDIT:
    audit_alignment(prices_df, carbon_df)

# ---------- 6) Save ----------
merged_df = merged_df.drop(columns=["timestamp"])

# reorder columns nicely
cols = [
    "trade_ts", "trade_date",
    "delivery_ts", "delivery_date",
    "tau",
    "price_gbp_mwh",
    "carbon_gco2_kwh",
]
merged_df = merged_df[cols]

merged_df.to_parquet("data/training_data.parquet", index=False)