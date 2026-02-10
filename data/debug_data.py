import eikon as ek, configparser as cp
from pathlib import Path
import pandas as pd

cfg = cp.ConfigParser()
cfg.read(Path("data") / "eikon.cfg")
ek.set_app_key(cfg["eikon"]["app_id"].strip())

ric = "EPXGBAUCD1H01H1"
ts_short = ek.get_timeseries(ric, start_date="2021-01-01", end_date="2021-01-10", fields=["CLOSE"])
ts_long  = ek.get_timeseries(ric, start_date="2021-01-01", end_date="2023-01-01", fields=["CLOSE"])

print("short:", None if ts_short is None else ts_short.shape)
print("long :", None if ts_long  is None else ts_long.shape)
print(ts_long.head() if ts_long is not None else None)