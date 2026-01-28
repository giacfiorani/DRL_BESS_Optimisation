from __future__ import annotations
import os, time, datetime as dt
from typing import Any, Dict, List, Optional, Sequence
import requests
import pandas as pd
import eikon as ek
import configparser as cp
from eikon_rics_lists import DA_HH_RICS, SSP_RICS #import rics list for Wholesale prices 

# all data date timezones are already in UTC 

# =======
# ----- SYSTEM PRICE DATA -----
# ======

##
# -- EIKON CONNECTION ---
# Day Ahead Prices from Refinitiv Workspace
# System Buy and Sell Prices (SSP & SBP)
# ===
def fetch_48sp_curve(
    start_date: str,
    end_date: str,
    rics: list[str],
    *,
    curve_name: str,          # e.g. "DA", "SSP", "SBP"
    field: str = "CLOSE",
    tz_naive: bool = True,
) -> pd.DataFrame:
    """
    Fetch a 48-settlement-period daily curve from Eikon.

    Returns LONG dataframe with:
      delivery_date (date), settlement_period (1..48), delivery_ts, trade_ts,
      trade_date, price_gbp_mwh, curve_name

    Important:
      - Index returned by Eikon is assumed to be daily delivery date.
      - We do NOT localize to UTC; we just normalize to midnight and then build SP timestamps.
      - 'trade_ts' here is set to delivery_ts - 1 day
    """
    if len(rics) != 48:
        raise ValueError(f"{curve_name}: expected 48 RICs, got {len(rics)}")

    cfg = cp.ConfigParser()
    cfg.read("eikon.cfg")
    ek.set_app_key(cfg["eikon"]["app_id"])

    parts = [] # fetch each SP seperatly 

    #loop through the 48 rics and align it to its settlement period (sp) number
    for sp, ric in enumerate(rics, start=1):
        #dataframe indexed by date
        ts = ek.get_timeseries(
            rics=ric,
            start_date=start_date,
            end_date=end_date,
            fields=[field],
        )
        if ts is None or len(ts) == 0:
            raise RuntimeError(f"Eikon returned no data for {ric} in [{start_date}, {end_date}]")

        ts = ts.sort_index() #ensures dataframe dates are stored ascending

        # column name handling - if column not called CLOSE take the first column
        col = field if field in ts.columns else ts.columns[0]
        s = ts[col].rename("price_gbp_mwh").to_frame()

        s["settlement_period"] = sp
        s["ric"] = ric
        parts.append(s)

    # Stack all SP blocks vertically.
    wide = pd.concat(parts, axis=0)
    wide.index = pd.to_datetime(wide.index).normalize() # convert to datetime
    wide.index.name = "delivery_date"

   #sort sp so its in order fromm 1 to 48
    df = wide.reset_index().sort_values(["delivery_date", "settlement_period"]).reset_index(drop=True) 

    # Build timestamps (clock-day: SP1=00:00, SP48=23:30)
    df["delivery_ts"] = df["delivery_date"] + pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="min")
    df["trade_ts"] = df["delivery_ts"] - pd.Timedelta(days=1)
    df["trade_date"] = df["trade_ts"].dt.floor("D")

    # Stable timestamp = delivery half-hour
    df["timestamp"] = df["delivery_ts"]

    df["curve_name"] = curve_name

    if tz_naive:
        for c in ["timestamp", "delivery_ts", "trade_ts", "delivery_date", "trade_date"]:
            df[c] = pd.to_datetime(df[c]).dt.tz_localize(None)

    return df[
        [
            "timestamp",
            "delivery_ts",
            "trade_ts",
            "delivery_date",
            "trade_date",
            "settlement_period",
            "price_gbp_mwh",
            "curve_name",
            "ric",
        ]
    ]

# ====
# Intraday Prices from Elexon
# ====

BASE = "https://data.elexon.co.uk/bmrs/api/v1"

def _headers() -> Dict[str, str]:
    """Use an API key if you have one (optional)."""
    key = os.getenv("ELEXON_API_KEY", "").strip()
    h = {"Accept": "application/json"}
    if key:
        h["x-api-key"] = key
    return h

# JSON get request function that handles rate limits
def _get_json(url: str, params: Dict[str, Any], timeout: int = 30, max_retries: int = 8) -> Any:
    """GET with 429 backoff."""
    #try request up to 8 times
    backoff = 1.0 
    for attempt in range(max_retries):
        r = requests.get(url, headers=_headers(), params=params, timeout=timeout)
        if r.status_code == 200: # success -> returned parsed JSON 
            return r.json()
        if r.status_code == 429: # too many requests -> wait and retry with exponential backoff.
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 15.0)
            continue
        if r.status_code in (400, 404): # If request is invalid or there’s no data → just return None rather than crashing.
            return None
        r.raise_for_status()
    raise RuntimeError(f"Max retries exceeded for {url} with params={params}")


def sp_to_halfhour_start(settlement_date: str, sp: int, tz: str = "Europe/London") -> pd.Timestamp:
    """
    Elexon settlement periods are half-hours. This builds the HH start timestamp.
    NOTE: On DST days there can be 46 or 50 periods. Elexon handles it via sp range.
    """
    # Treat settlement_date as local clock day; keep as timezone-aware for safety
    d = pd.Timestamp(settlement_date).tz_localize(tz)
    ts = d + pd.to_timedelta((sp - 1) * 30, unit="min")
    return ts

# Fetches Market Index Data (MID), which are Intraday Prices
def fetch_market_index(
    from_dt: str,
    to_dt: str,
    data_providers: Optional[Sequence[str]] = None,
    settlementPeriodFrom: Optional[int] = None,
    settlementPeriodTo: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetch Elexon Market Index Data (MID) time series.

    Parameters
    ----------
    from_dt, to_dt : RFC3339 datetime strings (e.g. "2022-06-01T00:00Z")
    data_providers : list like ["N2EXMIDP"] or ["APXMIDP"] or both; if None, fetch both. 
    settlementPeriodFrom/To : optional int 1..50; if provided, from/to are treated as settlement dates (time ignored).
    """
    url = f"{BASE}/balancing/pricing/market-index" # endpoint

    params: Dict[str, Any] = {"from": from_dt, "to": to_dt, "format": "json"}
    if settlementPeriodFrom is not None:
        params["settlementPeriodFrom"] = int(settlementPeriodFrom)
    if settlementPeriodTo is not None:
        params["settlementPeriodTo"] = int(settlementPeriodTo)

    # Elexon expects repeated query args for arrays. requests supports list values.
    if data_providers:
        params["dataProviders"] = list(data_providers)

    #extracts the list of rows
    payload = _get_json(url, params=params)
    if payload is None: 
        return pd.DataFrame()

    # Most BMRS endpoints wrap rows in {"data":[...]}
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    if not rows: # no rows then emptuy dataframe
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Normalize typical column names (Elexon may use different names/casing)
    # API can return marketIndexPrice/marketIndexVolume or price/volume
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "settlementdate":
            rename_map[c] = "settlementDate"
        elif lc == "settlementperiod":
            rename_map[c] = "settlementPeriod"
        elif lc == "dataprovider":
            rename_map[c] = "dataProvider"
        elif lc == "marketindexprice" or lc == "price":
            rename_map[c] = "marketIndexPrice"
        elif lc == "marketindexvolume" or lc == "volume":
            rename_map[c] = "marketIndexVolume"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Build timestamp from settlementDate + settlementPeriod
    if "settlementDate" in df.columns and "settlementPeriod" in df.columns:
        df["settlementDate"] = pd.to_datetime(df["settlementDate"]).dt.date.astype(str)
        df["settlementPeriod"] = df["settlementPeriod"].astype(int)

        #build local London timestamp for sp
        df["timestamp"] = [
            sp_to_halfhour_start(d, sp).tz_convert("UTC")
            for d, sp in zip(df["settlementDate"], df["settlementPeriod"])
        ]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Sort + tidy
    sort_cols = [c for c in ["timestamp", "dataProvider", "settlementDate", "settlementPeriod"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Keep a nice front-of-table order; keep other fields after
    front = [c for c in [
        "timestamp",
        "settlementDate",
        "settlementPeriod",
        "dataProvider",
        "marketIndexPrice",
        "marketIndexVolume",
    ] if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]

    return df

def fetch_market_index_range_by_settlement_date(
    start_date: str,
    end_date: str,
    data_providers: Optional[Sequence[str]] = None,
    sleep_s: float = 0.15,
) -> pd.DataFrame:
    """
    Convenience wrapper if you want to fetch by settlement date range.
    Uses from/to as date filters (time ignored) by providing settlementPeriodFrom/To.

    start_date/end_date: 'YYYY-MM-DD' inclusive
    """
    #turn string to dateformat
    start = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)

    #iterate over each day
    dfs: List[pd.DataFrame] = []
    d = start
    while d <= end:
        # Query one day window (settlement date filtering makes time irrelevant)
        from_dt = f"{d.isoformat()}T00:00Z"
        to_dt = f"{(d + dt.timedelta(days=1)).isoformat()}T00:00Z"
        df_day = fetch_market_index(
            from_dt=from_dt,
            to_dt=to_dt,
            data_providers=data_providers,
            settlementPeriodFrom=1,
            settlementPeriodTo=50,
        )
        if not df_day.empty:
            dfs.append(df_day)
        time.sleep(sleep_s) # small sleep to not hammer API
        d += dt.timedelta(days=1)
    
    #stack all days
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()



#=========
# ----- CARBON INTENSITY DATA -----
#=======

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

# ----- Demand Data - Actual Load ----- DONT KEEP FOR NOW

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