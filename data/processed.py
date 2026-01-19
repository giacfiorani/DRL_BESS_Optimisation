import pandas as pd
from data_loader import fetch_system_prices, fetch_carbon_sql
from data_audit import audit_alignment


start = "2021-01-01"
end   = "2023-01-01"

prices_df = fetch_system_prices(start, end)
carbon_df = fetch_carbon_sql(start, end)

# ---------- 1) Clean timestamps ----------
def clean_ts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df = df.sort_values("timestamp")
    df = df.drop_duplicates("timestamp")
    df = df.reset_index(drop=True)
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

# Keep only relevant columns
prices_df = prices_df.loc[:, ["timestamp", "price_gbp_mwh"]]
carbon_df = carbon_df.loc[:, ["timestamp", "carbon_gco2_kwh"]]

# ---------- 3) Merge ----------
merged_df = prices_df.merge(
    carbon_df[["timestamp", "carbon_gco2_kwh"]],
    on="timestamp",
    how="inner",
)

# ---------- 4) Add tau per day ----------
# timestamp is already datetime from clean_ts, so no need to reconvert
merged_df["date"] = merged_df["timestamp"].dt.date
merged_df = merged_df.sort_values("timestamp").reset_index(drop=True)

merged_df["tau"] = ((
    merged_df["timestamp"].dt.hour * 2
    + (merged_df["timestamp"].dt.minute // 30)+1)
)

# -----  5) Data Audit -----
RUN_AUDIT = False  

if RUN_AUDIT:
    audit_alignment(prices_df, carbon_df)

# ---------- 6) Save to parquet ----------
merged_df.to_parquet("data/training_data.parquet", index=False)