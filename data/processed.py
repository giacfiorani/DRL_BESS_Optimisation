from __future__ import annotations

from pathlib import Path
import pandas as pd

from data_loader import (
    fetch_48sp_curve,
    fetch_market_index_range_by_settlement_date,
    fetch_uka_daily,
    fetch_ci_forecast,
    fetch_carbon_sql,
    compute_mef_model_a,
)
from eikon_rics_lists import DA_HH_RICS

# -----------------------
# Config
# -----------------------
START = "2022-01-01"
END   = "2026-01-01"

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "data.parquet"

# -----------------------
# Helpers
# -----------------------
def clean_ts_utc_naive(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Parse timestamp to tz-naive UTC and sort."""
    df = df.copy()
    if ts_col not in df.columns:
        raise KeyError(f"Expected column '{ts_col}' in df columns={df.columns.tolist()}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).reset_index(drop=True)
    return df


def dedup_last(df: pd.DataFrame, subset: list[str]) -> pd.DataFrame:
    """Sort by subset[0] if it's timestamp-like then keep last duplicate."""
    df = df.copy()
    df = df.drop_duplicates(subset=subset, keep="last").reset_index(drop=True)
    return df


def to_delivery_hh_key(ts: pd.Series) -> pd.Series:
    """
    Canonical delivery half-hour key (tz-naive).
    Force rounding/flooring to 30 minutes so merges are stable.
    """
    ts = pd.to_datetime(ts, errors="coerce")
    return ts.dt.floor("30min")


# -----------------------
# 1) Pull datasets
# -----------------------
print("Pulling DA 48SP curve...")
da_df = fetch_48sp_curve(START, END, DA_HH_RICS, curve_name="DA")

print("Pulling MID market index...")
mid_df = fetch_market_index_range_by_settlement_date(
    start_date=START,
    end_date=END,
    data_providers=["APXMIDP"],
)

print("Pulling UKA daily...")
uka_df = fetch_uka_daily(START, END)

print("Pulling CI forecast (forecast-only)...")
ci_fc_df = fetch_ci_forecast(START, END, tz_naive=True, target_freq="30min")

print("Pulling CI actual (historic realised)...")
ci_act_df = fetch_carbon_sql(START, END)

print("Computing MEF Model A...")
mef_df = compute_mef_model_a(START, END, tz_naive=True)


# -----------------------
# 2) Clean timestamps + standardise keys
# -----------------------
# DA already has delivery_ts/trade_ts etc, but we still normalise.
da_df = da_df.copy()
for col in ["timestamp", "delivery_ts", "trade_ts"]:
    if col in da_df.columns:
        da_df[col] = pd.to_datetime(da_df[col], utc=True, errors="coerce").dt.tz_convert(None)

# Prefer delivery_ts as our canonical join key.
if "delivery_ts" not in da_df.columns:
    # fallback: DA implementations sometimes return timestamp only
    if "timestamp" not in da_df.columns:
        raise KeyError("DA dataframe missing both 'delivery_ts' and 'timestamp'.")
    da_df["delivery_ts"] = to_delivery_hh_key(da_df["timestamp"])
else:
    da_df["delivery_ts"] = to_delivery_hh_key(da_df["delivery_ts"])

# MID is timestamp-based
mid_df = clean_ts_utc_naive(mid_df, "timestamp")
mid_df["delivery_ts"] = to_delivery_hh_key(mid_df["timestamp"])

# CI forecast: timestamp-based already
ci_fc_df = clean_ts_utc_naive(ci_fc_df, "timestamp")
ci_fc_df["delivery_ts"] = to_delivery_hh_key(ci_fc_df["timestamp"])

# CI actual: timestamp-based
ci_act_df = clean_ts_utc_naive(ci_act_df, "timestamp")
ci_act_df["delivery_ts"] = to_delivery_hh_key(ci_act_df["timestamp"])

# MEF: timestamp-based
mef_df = clean_ts_utc_naive(mef_df, "timestamp")
mef_df["delivery_ts"] = to_delivery_hh_key(mef_df["timestamp"])

# UKA: daily; we merge on delivery_date
uka_df = clean_ts_utc_naive(uka_df, "timestamp")
uka_df["delivery_date"] = uka_df["timestamp"].dt.floor("D")


# -----------------------
# 3) Rename columns to canonical names
# -----------------------
# DA
if "price_gbp_mwh" in da_df.columns:
    da_df = da_df.rename(columns={"price_gbp_mwh": "da_price_gbp_mwh"})
elif "da_price_gbp_mwh" not in da_df.columns:
    raise KeyError(f"DA missing price column. columns={da_df.columns.tolist()}")

# MID
if "marketIndexPrice" not in mid_df.columns:
    raise KeyError(f"MID missing marketIndexPrice. columns={mid_df.columns.tolist()}")
mid_df = mid_df.rename(columns={"marketIndexPrice": "mid_price_gbp_mwh"})

# CI forecast
if "ci_forecast_gco2_kwh" not in ci_fc_df.columns:
    raise KeyError(f"CI forecast missing ci_forecast_gco2_kwh. columns={ci_fc_df.columns.tolist()}")

# CI actual
if "ci_actual_gco2_kwh" not in ci_act_df.columns:
    raise KeyError(f"CI actual missing carbon_gco2_kwh. columns={ci_act_df.columns.tolist()}")
ci_act_df = ci_act_df.rename(columns={"ci_actual_gco2_kwh": "ci_actual_gco2_kwh"})

# MEF
if "mef_gco2_kwh" not in mef_df.columns:
    raise KeyError(f"MEF missing mef_gco2_kwh. columns={mef_df.columns.tolist()}")

# UKA
if "uka_gbp_tco2" not in uka_df.columns:
    raise KeyError(f"UKA missing uka_gbp_tco2. columns={uka_df.columns.tolist()}")


# -----------------------
# 4) De-duplicate on join keys
# -----------------------
da_df     = dedup_last(da_df, ["delivery_ts"])
mid_df    = dedup_last(mid_df, ["delivery_ts"])
ci_fc_df  = dedup_last(ci_fc_df, ["delivery_ts"])
ci_act_df = dedup_last(ci_act_df, ["delivery_ts"])
mef_df    = dedup_last(mef_df, ["delivery_ts"])

uka_daily = (
    uka_df.sort_values("timestamp")
          .drop_duplicates("delivery_date", keep="last")
          .loc[:, ["delivery_date", "uka_gbp_tco2"]]
          .reset_index(drop=True)
)


# -----------------------
# 5) Keep only needed columns
# -----------------------
da_keep = [
    "delivery_ts",
    "da_price_gbp_mwh",
]
# keep trade_ts/trade_date/delivery_date/tau if your DA loader already provides them;
# otherwise we’ll regenerate later.
optional_da_cols = ["trade_ts", "trade_date", "delivery_date", "settlement_period", "tau"]
for c in optional_da_cols:
    if c in da_df.columns:
        da_keep.append(c)

da_df = da_df[da_keep].copy()

mid_df = mid_df[["delivery_ts", "mid_price_gbp_mwh"]].copy()
ci_df  = ci_fc_df[["delivery_ts", "ci_forecast_gco2_kwh"]].merge(
    ci_act_df[["delivery_ts", "ci_actual_gco2_kwh"]],
    on="delivery_ts",
    how="left",
)
mef_df = mef_df[["delivery_ts", "mef_gco2_kwh"]].copy()


# -----------------------
# 6) Merge everything on delivery_ts
# -----------------------
merged = (
    da_df
    .merge(mid_df, on="delivery_ts", how="left")
    .merge(ci_df,  on="delivery_ts", how="left")
    .merge(mef_df, on="delivery_ts", how="left")
    .sort_values("delivery_ts")
    .reset_index(drop=True)
)

merged["delivery_date"] = pd.to_datetime(merged["delivery_ts"]).dt.floor("D")

merged = merged.merge(uka_daily, on="delivery_date", how="left")


# -----------------------
# 7) Canonical fields (trade_ts/trade_date/tau) if missing
# -----------------------
# tau = settlement period 1..48 from delivery_ts
if "tau" not in merged.columns:
    merged["tau"] = (
        merged["delivery_ts"].dt.hour * 2
        + (merged["delivery_ts"].dt.minute // 30)
        + 1
    ).astype(int)

# trade_ts = delivery_ts - 1 day (simple “publish on D-1” convention)
if "trade_ts" not in merged.columns:
    merged["trade_ts"] = merged["delivery_ts"] - pd.Timedelta(days=1)

if "trade_date" not in merged.columns:
    merged["trade_date"] = merged["trade_ts"].dt.floor("D")


# -----------------------
# 8) Fill missing values
# -----------------------
# MID prices often have small gaps; forward/back fill is acceptable for training stability.
# CI forecast can be ffilled (it’s a forecast series); actual CI should NOT be ffilled too aggressively,
# but for env continuity, we fill small gaps after merge.
fill_cols = [
    "mid_price_gbp_mwh",
    "ci_forecast_gco2_kwh",
    "ci_actual_gco2_kwh",
    "mef_gco2_kwh",
    "uka_gbp_tco2",
]
for col in fill_cols:
    if col in merged.columns:
        merged[col] = merged[col].ffill().bfill()

# Drop any remaining critical nulls
merged = merged.dropna(
    subset=[
        "da_price_gbp_mwh",
        "mid_price_gbp_mwh",
        "ci_forecast_gco2_kwh",
        "ci_actual_gco2_kwh",
        "uka_gbp_tco2",
        "mef_gco2_kwh",
    ]
).reset_index(drop=True)


# -----------------------
# 9) Ensure one row per (delivery_date, tau) and sorted
# -----------------------
cols = [
    "trade_ts",
    "trade_date",
    "delivery_ts",
    "delivery_date",
    "tau",
    "da_price_gbp_mwh",
    "mid_price_gbp_mwh",
    "ci_forecast_gco2_kwh",
    "ci_actual_gco2_kwh",
    "mef_gco2_kwh",
    "uka_gbp_tco2",
]

merged = merged[cols].sort_values(["delivery_ts"]).reset_index(drop=True)
merged = merged.drop_duplicates(subset=["delivery_date", "tau"], keep="first").reset_index(drop=True)

# sanity: tau in 1..48
bad_tau = (~merged["tau"].between(1, 48)).mean()
if bad_tau > 0:
    raise ValueError("Found tau outside 1..48 after build.")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
merged.to_parquet(OUT_PATH, index=False)

print("Saved:", len(merged), "rows ->", str(OUT_PATH))
print("Range:", merged["delivery_ts"].min(), "to", merged["delivery_ts"].max())
print("Null %:", merged.isna().mean().sort_values(ascending=False).head(10).to_dict())