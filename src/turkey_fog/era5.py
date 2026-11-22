from pathlib import Path

import numpy as np
import pandas as pd

ERA5_FEATURES = ["temperature_2m", "dpd", "t2m_minus_skin", "wind10"]


def download_era5land(cfg):
    """Hourly ERA5-Land values at each station from Google Earth Engine,
    one CSV per station and year. Requests are made month by month to stay
    under the Earth Engine memory limit. Existing files are skipped."""
    import ee
    e = cfg["era5"]
    ee.Initialize(project=e["gee_project"])
    coll = ee.ImageCollection(e["collection"]).select(e["bands"])
    out = Path(cfg["paths"]["era5_raw"])
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = cfg["years"]
    for st, p in cfg["stations"].items():
        point = ee.Geometry.Point(p["lon"], p["lat"])
        for year in range(y0, y1 + 1):
            f = out / f"{st}_{year}.csv"
            if f.exists():
                continue
            parts = []
            for m in range(1, 13):
                start = pd.Timestamp(year, m, 1)
                end = start + pd.offsets.MonthBegin(1)
                rows = coll.filterDate(str(start.date()), str(end.date())).getRegion(point, e["scale"]).getInfo()
                parts.append(pd.DataFrame(rows[1:], columns=rows[0]))
            pd.concat(parts).to_csv(f, index=False)
            print(st, year, "downloaded", flush=True)


def load_era5land(cfg):
    """All downloaded ERA5-Land files as one table with the derived fog features:
    dpd = 2 m temperature − dew point (near 0 = saturated air),
    t2m_minus_skin = 2 m temperature − surface temperature (positive = surface cooling),
    wind10 = 10 m wind speed."""
    files = sorted(Path(cfg["paths"]["era5_raw"]).glob("*.csv"))
    era = pd.concat(pd.read_csv(f).assign(station=f.stem.split("_")[0]) for f in files)
    era["hour"] = pd.to_datetime(era["time"], unit="ms")
    era["dpd"] = era["temperature_2m"] - era["dewpoint_temperature_2m"]
    era["t2m_minus_skin"] = era["temperature_2m"] - era["skin_temperature"]
    era["wind10"] = np.hypot(era["u_component_of_wind_10m"], era["v_component_of_wind_10m"])
    return era[["station", "hour"] + ERA5_FEATURES]


def add_era5(df, era):
    """Attach ERA5-Land values from the nearest full hour of each sample."""
    df = df.assign(hour=df["valid"].dt.round("h"))
    return df.merge(era, on=["station", "hour"], how="left").drop(columns="hour")
