from __future__ import annotations
import os, time, datetime as dt
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
import eikon as ek
import configparser as cp
from eikon_rics_lists import DA_HH_RICS #import rics list for Wholesale prices 

# all data date timezones are already in UTC 

# ----- SYSTEM PRICE DATA -----

# -- EIKON CONNECTION ---

def fetch_system_prices(start_date: str, end_date: str) -> pd.DataFrame:
    cfg = cp.ConfigParser()
    cfg.read("eikon.cfg")

    ek.set_app_key(cfg["eikon"]["app_id"])

    parts = {} #create empty price data dictionary - key=ric & value = time_series
    sp_to_ric = {} #associate rics to the settlement period  (i.e. sp01)

    #since when doing it all in once, creates merging issues, we loop through the ric to output its timeseries one at a time
    for i, r in enumerate(DA_HH_RICS, start=1):
        sp = f"sp_{i:02d}" 
        ts = ek.get_timeseries(rics=r, 
        start_date=start_date,
        end_date=end_date,
        fields=["CLOSE"]).sort_index()

        #set columns to settlement periods (sp01,...)
        col = "CLOSE" if "CLOSE" in ts.columns else ts.columns[0]
        ts = ts.rename(columns={col: sp})

        parts[sp] = ts
        sp_to_ric[sp] = r

    #Concatenate the dictionary 
    price_data = pd.concat([parts[sp] for sp in sorted(parts.keys())], axis=1).sort_index()
    #dropping incomplete days  
    complete_price_data= price_data.dropna(how="any")

    # make delivery_date an explicit column for melt
    complete_price_data.index.name = "delivery_date"
    wide = complete_price_data.reset_index()

    # wide -> long
    wholesale_data = wide.melt(
        id_vars=["delivery_date"],
        var_name="sp",
        value_name="price_gbp_mwh"
    ).sort_values(["delivery_date", "sp"]).reset_index(drop=True)

    # sp_01 -> 1, ..., sp_48 -> 48
    wholesale_data["settlement_period"] = wholesale_data["sp"].str.extract(r"(\d+)").astype(int)

    #create timestamp column : day + hour + minutes in UTC timezone
    wholesale_data["timestamp"] = (
        pd.to_datetime(wholesale_data["delivery_date"], utc=True)
        + pd.to_timedelta((wholesale_data["settlement_period"] - 1) * 30, unit="min")
    )

    return wholesale_data

# ----- Carbon Intensity Data -----

URL = "https://api.neso.energy/api/3/action/datastore_search_sql"
RESOURCE_ID = "f93d1835-75bc-43e5-84ad-12472b180a98"  # Carbon intensity dataset ID


def fetch_carbon_sql(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch carbon intensity rows where "from" is in [start_date, end_date),
    returned as a tidy DataFrame with columns: timestamp, carbon_gco2_kwh.
    Dates are ISO strings: 'YYYY-MM-DD'.
    """
    sql = f"""
        SELECT *
        FROM "{RESOURCE_ID}"
        WHERE "DATETIME" >= '{start_date}T00:00:00+00:00'
        AND "DATETIME" <  '{end_date}T00:00:00+00:00'
        ORDER BY "DATETIME" ASC
    """
    r = requests.get(URL, params={"sql": sql}, timeout=60)
    r.raise_for_status()
    records = r.json()["result"]["records"]
    df = pd.DataFrame(records)

    if df.empty:
        return df

    # Normalise columns (schema can vary slightly)
    df.columns = [c.lower() for c in df.columns]
    # Timestamp column (common candidates: "from", "datetime")
    ts_col = "from" if "from" in df.columns else "datetime"
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

    # Intensity column (common candidates)
    if "carbon_intensity" in df.columns:
        ci_col = "carbon_intensity"
    elif "intensity" in df.columns:
        ci_col = "intensity"
    elif "actual" in df.columns:
        ci_col = "actual"
    else:
        # fallback to any single numeric col if present
        num_cols = df.select_dtypes("number").columns
        if len(num_cols) == 1:
            ci_col = num_cols[0]
        else:
            raise KeyError(f"Can't find carbon intensity column in columns={df.columns.tolist()}")

    out = (df[["timestamp", ci_col]]
           .rename(columns={ci_col: "carbon_gco2_kwh"})
           .dropna(subset=["timestamp"])
           .sort_values("timestamp")
           .drop_duplicates("timestamp")
           .reset_index(drop=True))
    return out

# ----- Demand Data - Acutal Load ----- DONT KEEP FOR NOW

# def fetch_demand_window(
#     date_from: str,  # 'YYYY-MM-DD'
#     date_to: str,    # 'YYYY-MM-DD' (exclusive end or same-day for <=7d)
#     sp_from: Optional[int] = None,
#     sp_to: Optional[int] = None,
# ) -> pd.DataFrame:
#     """
#     Calls /demand/actual/total?from=YYYY-MM-DD&to=YYYY-MM-DD[&settlementPeriodFrom=..&settlementPeriodTo=..]
#     The API supports a max window of 7 days per request.
#     Returns columns: timestamp (datetime), settlementDate, settlementPeriod, demand_mw
#     """
#     url = f"{BASE}/demand/actual/total"
#     params: Dict[str, Any] = {"from": date_from, "to": date_to, "format": "json"}
#     if sp_from is not None:
#         params["settlementPeriodFrom"] = sp_from
#     if sp_to is not None:
#         params["settlementPeriodTo"] = sp_to

#     r = requests.get(url, headers=_headers(), params=params, timeout=30)
#     r.raise_for_status()
#     payload = r.json()

#     data = payload.get("data", [])
#     if not data:
#         return pd.DataFrame(columns=["timestamp","settlementDate","settlementPeriod","demand_mw"])

#     df = pd.DataFrame(data)
#     # Normalise columns
#     if "startTime" in df.columns:
#         ts = pd.to_datetime(df["startTime"], errors="coerce")
#     elif "timestamp" in df.columns:
#         ts = pd.to_datetime(df["timestamp"], errors="coerce")
#     else:
#         raise KeyError(f"No time column in demand payload: {df.columns.tolist()}")

#     qty_col = "quantity" if "quantity" in df.columns else None
#     if qty_col is None:
#         # fall back to the first numeric column if schema changes
#         nums = df.select_dtypes("number").columns.tolist()
#         if not nums:
#             raise KeyError("No numeric demand column found in demand payload")
#         qty_col = nums[0]

#     out = pd.DataFrame({
#         "timestamp": ts,
#         "settlementDate": df.get("settlementDate"),
#         "settlementPeriod": df.get("settlementPeriod"),
#         "demand_mw": pd.to_numeric(df[qty_col], errors="coerce"),
#     }).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

#     return out

# def fetch_demand_range(
#     start_date: str,  # 'YYYY-MM-DD'
#     end_date: str,    # 'YYYY-MM-DD' (inclusive end handled below)
#     sp_from: Optional[int] = None,
#     sp_to: Optional[int] = None,
# ) -> pd.DataFrame:
#     """
#     Fetches demand for an arbitrarily long period by chunking into <=7-day windows.
#     Returns tidy DataFrame with columns: timestamp, settlementDate, settlementPeriod, demand_mw
#     """
#     d0 = dt.date.fromisoformat(start_date)
#     d1 = dt.date.fromisoformat(end_date)
#     # Make d1 exclusive by adding 1 day when we form the final chunk bound
#     end_excl = d1 + dt.timedelta(days=1)

#     frames: List[pd.DataFrame] = []
#     chunk_start = d0
#     while chunk_start < end_excl:
#         chunk_end = min(chunk_start + dt.timedelta(days=7), end_excl)
#         df_chunk = fetch_demand_window(
#             chunk_start.isoformat(),
#             chunk_end.isoformat(),
#             sp_from=sp_from, sp_to=sp_to
#         )
#         if not df_chunk.empty:
#             frames.append(df_chunk)
#         chunk_start = chunk_end

#     if not frames:
#         return pd.DataFrame(columns=["timestamp","settlementDate","settlementPeriod","demand_mw"])

#     df = pd.concat(frames, ignore_index=True)
#     # De-duplicate just in case of overlapping edges
#     df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
#     return df