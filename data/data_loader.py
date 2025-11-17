from __future__ import annotations
import os, time, datetime as dt
from typing import List, Dict, Any, Optional
import requests
import pandas as pd

# ----- System Prices Data -----

BASE = "https://data.elexon.co.uk/bmrs/api/v1"

def _headers():
    """Use an API key if you have one (optional)."""
    key = os.getenv("ELEXON_API_KEY", "").strip()
    h = {"Accept": "application/json"}
    if key:
        h["x-api-key"] = key
    return h

def fetch_system_price(settlement_date: str, settlement_period: int) -> Dict[str, Any] | None:
    """
    Fetch one settlement period for a given date.
    settlement_date: 'YYYY-MM-DD'
    settlement_period: 1..48 (46/50 on clock-change days)
    Returns a dict (systemBuyPrice/systemSellPrice/...) or None if missing.
    """
    url = f"{BASE}/balancing/settlement/system-prices/{settlement_date}/{settlement_period}"
    r = requests.get(url, headers=_headers(), timeout=20)

    if r.status_code == 200:
        data = r.json()
        # Some endpoints return {"data":[...]} while others return a list or dict
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, list):
            return data[0] if data else None
        if isinstance(data, dict):
            return data
        return None
    if r.status_code in (400, 404):
        return None  # invalid/missing SP
    if r.status_code == 429:         # rate limit
        time.sleep(2.0)
        return fetch_system_price(settlement_date, settlement_period)
    r.raise_for_status()

def sp_to_halfhour_start(settlement_date: str, sp: int) -> pd.Timestamp:
    d = pd.to_datetime(settlement_date)
    return d + pd.to_timedelta((sp - 1) * 30, "m")

def fetch_system_prices_day(settlement_date: str) -> pd.DataFrame:
    """Fetch all SPs for one date (tries 1..50), return a tidy DataFrame."""
    rows: List[Dict[str, Any]] = []
    for sp in range(1, 51):  # handles 46/48/50 SP days
        rec = fetch_system_price(settlement_date, sp)
        if rec is None:
            continue
        rec["_settlementDate"] = settlement_date
        rec["_settlementPeriod"] = sp
        rec["timestamp"] = sp_to_halfhour_start(settlement_date, sp)
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp","_settlementDate","_settlementPeriod",
            "systemSellPrice","systemBuyPrice","netImbalanceVolume"
        ])

    df = pd.DataFrame(rows)

    # Short, consistent column names
    rename = {
        "systemSellPrice": "ssp",
        "systemBuyPrice":  "sbp",
        "netImbalanceVolume": "niv",
        "createdDateTime": "createdDateTime",
    }
    for k, v in rename.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    df = df.sort_values("timestamp").reset_index(drop=True)
    # Keep key fields first, then any totals the API returns
    return df[[
        "timestamp","_settlementDate","_settlementPeriod",
        *(c for c in ["ssp","sbp","niv","createdDateTime"] if c in df.columns),
        *[c for c in df.columns if c.startswith("total")]
    ]]

def fetch_system_prices_range(start_date: str, end_date: str, sleep_s: float = 0.1) -> pd.DataFrame:
    """Inclusive range 'YYYY-MM-DD' → merged DataFrame."""
    start = dt.date.fromisoformat(start_date)
    end   = dt.date.fromisoformat(end_date)
    dfs: List[pd.DataFrame] = []
    d = start
    while d <= end:
        df_day = fetch_system_prices_day(d.isoformat())
        if not df_day.empty:
            dfs.append(df_day)
        time.sleep(sleep_s)  # polite spacing
        d += dt.timedelta(days=1)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


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
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")

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

# ----- Demand Data - Acutal Load -----

def fetch_demand_window(
    date_from: str,  # 'YYYY-MM-DD'
    date_to: str,    # 'YYYY-MM-DD' (exclusive end or same-day for <=7d)
    sp_from: Optional[int] = None,
    sp_to: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calls /demand/actual/total?from=YYYY-MM-DD&to=YYYY-MM-DD[&settlementPeriodFrom=..&settlementPeriodTo=..]
    The API supports a max window of 7 days per request.
    Returns columns: timestamp (datetime), settlementDate, settlementPeriod, demand_mw
    """
    url = f"{BASE}/demand/actual/total"
    params: Dict[str, Any] = {"from": date_from, "to": date_to, "format": "json"}
    if sp_from is not None:
        params["settlementPeriodFrom"] = sp_from
    if sp_to is not None:
        params["settlementPeriodTo"] = sp_to

    r = requests.get(url, headers=_headers(), params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()

    data = payload.get("data", [])
    if not data:
        return pd.DataFrame(columns=["timestamp","settlementDate","settlementPeriod","demand_mw"])

    df = pd.DataFrame(data)
    # Normalise columns
    if "startTime" in df.columns:
        ts = pd.to_datetime(df["startTime"], errors="coerce")
    elif "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        raise KeyError(f"No time column in demand payload: {df.columns.tolist()}")

    qty_col = "quantity" if "quantity" in df.columns else None
    if qty_col is None:
        # fall back to the first numeric column if schema changes
        nums = df.select_dtypes("number").columns.tolist()
        if not nums:
            raise KeyError("No numeric demand column found in demand payload")
        qty_col = nums[0]

    out = pd.DataFrame({
        "timestamp": ts,
        "settlementDate": df.get("settlementDate"),
        "settlementPeriod": df.get("settlementPeriod"),
        "demand_mw": pd.to_numeric(df[qty_col], errors="coerce"),
    }).dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    return out

def fetch_demand_range(
    start_date: str,  # 'YYYY-MM-DD'
    end_date: str,    # 'YYYY-MM-DD' (inclusive end handled below)
    sp_from: Optional[int] = None,
    sp_to: Optional[int] = None,
) -> pd.DataFrame:
    """
    Fetches demand for an arbitrarily long period by chunking into <=7-day windows.
    Returns tidy DataFrame with columns: timestamp, settlementDate, settlementPeriod, demand_mw
    """
    d0 = dt.date.fromisoformat(start_date)
    d1 = dt.date.fromisoformat(end_date)
    # Make d1 exclusive by adding 1 day when we form the final chunk bound
    end_excl = d1 + dt.timedelta(days=1)

    frames: List[pd.DataFrame] = []
    chunk_start = d0
    while chunk_start < end_excl:
        chunk_end = min(chunk_start + dt.timedelta(days=7), end_excl)
        df_chunk = fetch_demand_window(
            chunk_start.isoformat(),
            chunk_end.isoformat(),
            sp_from=sp_from, sp_to=sp_to
        )
        if not df_chunk.empty:
            frames.append(df_chunk)
        chunk_start = chunk_end

    if not frames:
        return pd.DataFrame(columns=["timestamp","settlementDate","settlementPeriod","demand_mw"])

    df = pd.concat(frames, ignore_index=True)
    # De-duplicate just in case of overlapping edges
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df