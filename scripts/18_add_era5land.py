from pathlib import Path
import numpy as np
import pandas as pd

era = pd.concat(pd.read_csv(f).assign(station=f.stem.split("_")[0])
                for f in Path("data/raw/era5land").glob("*.csv"))
era["hour"] = pd.to_datetime(era["time"], unit="ms")
era["dpd"] = era["temperature_2m"] - era["dewpoint_temperature_2m"]
era["t2m_minus_skin"] = era["temperature_2m"] - era["skin_temperature"]
era["wind10"] = np.hypot(era["u_component_of_wind_10m"], era["v_component_of_wind_10m"])
era = era[["station", "hour", "temperature_2m", "dpd", "t2m_minus_skin", "wind10"]]

for name in ["features_multi", "features"]:
    df = pd.read_csv(f"data/processed/{name}.csv", parse_dates=["valid"])
    if "station" not in df:
        df["station"] = "LTAC"
    df["hour"] = df["valid"].dt.round("h")
    df = df.merge(era, on=["station", "hour"], how="left").drop(columns="hour")
    df.to_csv(f"data/processed/{name}_era5.csv", index=False)
    print(name, df.shape, "missing:", int(df[["dpd", "wind10"]].isna().sum().sum()))
    print(df.groupby("label")[["dpd", "t2m_minus_skin", "wind10"]].median().round(2))