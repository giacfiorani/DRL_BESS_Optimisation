from data_loader import fetch_demand_range, fetch_fuelhh_range, compute_mef_load_following_from_fuelhh

start, end = "2022-01-01", "2024-01-01"

demand_df = fetch_demand_range(start, end)

fuelhh_df = fetch_fuelhh_range(start, end, chunk_days=7, sleep_s=0.2, prefer_insights=True)
print("FUELHH shape:", fuelhh_df.shape, "cols sample:", fuelhh_df.columns[:10])

mef_df = compute_mef_load_following_from_fuelhh(fuelhh_df, demand_df)
print(mef_df.head(), "MEF null %:", mef_df["mef_gco2_kwh"].isna().mean())

mef_df.to_parquet("data/mef.parquet", index=False)