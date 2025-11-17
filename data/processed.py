import pandas as pd
import numpy as np

from data_loader import fetch_system_price, fetch_system_prices_day, fetch_system_prices_range, fetch_carbon_sql, fetch_demand_window, fetch_demand_range

start = "2024-01-01"
end   = "2024-03-01"

prices_df = fetch_system_prices_range(start, end)
carbon_df = fetch_carbon_sql(start, end)

print(prices_df.head())
print(carbon_df.head())
print(prices_df["timestamp"].min(), prices_df["timestamp"].max())
print(carbon_df["timestamp"].min(), carbon_df["timestamp"].max())

#Cleaning the timestamp column 
def  clean_ts(df):
    df=df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"],utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert(None)
    df = df.sort_values("timestamp")
    df=df.drop_duplicates("timestamp")
    df=df.reset_index(drop=True)
    return df

#clean both dataframes
prices_df = clean_ts(prices_df)
carbon_df = clean_ts(carbon_df)

start_common = max(prices_df["timestamp"].min(), carbon_df["timestamp"].min())
end_common = min(prices_df["timestamp"].max(), carbon_df["timestamp"].max())

#Filter both dataset to the above start and end common window
def clip(df, start, end):
    m= (df["timestamp"]>=start) & (df["timestamp"]<=end)
    return df.loc[m].reset_index(drop=True)

prices_df = clip(prices_df, start_common, end_common)
carbon_df = clip(carbon_df, start_common, end_common)

prices_df = prices_df.loc[:,["timestamp", "ssp", "sbp", "niv"]]
carbon_df = carbon_df.loc[:, ["timestamp", "carbon_gco2_kwh"]]

#merging both datasets into 1 
merged_df = prices_df.merge(carbon_df[["timestamp", "carbon_gco2_kwh"]],on="timestamp", how="inner")

#convert dataframe to parquet
merged_df.to_parquet("data/merged_data.parquet", index=False)