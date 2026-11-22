from pathlib import Path

import numpy as np
import pandas as pd

ERA5_FEATURES = ["temperature_2m", "dpd", "t2m_minus_skin", "wind10"]


def derive_features(raw):
    """Fog features from raw ERA5-Land columns. Used for training AND prediction,
    so both see exactly the same calculation:
    dpd = 2 m temperature − dew point (near 0 = saturated air),
    t2m_minus_skin = 2 m temperature − surface temperature (positive = surface cooling),
    wind10 = 10 m wind speed."""
    out = raw.copy()
    out["dpd"] = out["temperature_2m"] - out["dewpoint_temperature_2m"]
    out["t2m_minus_skin"] = out["temperature_2m"] - out["skin_temperature"]
    out["wind10"] = np.hypot(out["u_component_of_wind_10m"], out["v_component_of_wind_10m"])
    return out


def _region(cfg, coll, lon, lat, start, end):
    import ee
    rows = coll.filterDate(start, end).getRegion(ee.Geometry.Point(lon, lat), cfg["era5"]["scale"]).getInfo()
    return pd.DataFrame(rows[1:], columns=rows[0])


def _collection(cfg):
    import ee
    e = cfg["era5"]
    ee.Initialize(project=e["gee_project"])
    return ee.ImageCollection(e["collection"]).select(e["bands"])


def download_era5land(cfg):
    """Hourly ERA5-Land values at each station from Google Earth Engine,
    one CSV per station and year. Requests are made month by month to stay
    under the Earth Engine memory limit. Existing files are skipped."""
    coll = _collection(cfg)
    out = Path(cfg["paths"]["era5_raw"])
    out.mkdir(parents=True, exist_ok=True)
    y0, y1 = cfg["years"]
    for st, p in cfg["stations"].items():
        for year in range(y0, y1 + 1):
            f = out / f"{st}_{year}.csv"
            if f.exists():
                continue
            parts = []
            for m in range(1, 13):
                start = pd.Timestamp(year, m, 1)
                end = start + pd.offsets.MonthBegin(1)
                parts.append(_region(cfg, coll, p["lon"], p["lat"], str(start.date()), str(end.date())))
            pd.concat(parts).to_csv(f, index=False)
            print(st, year, "downloaded", flush=True)


def load_era5land(cfg):
    """All downloaded ERA5-Land files as one table with the derived fog features."""
    files = sorted(Path(cfg["paths"]["era5_raw"]).glob("*.csv"))
    era = pd.concat(pd.read_csv(f).assign(station=f.stem.split("_")[0]) for f in files)
    era["hour"] = pd.to_datetime(era["time"], unit="ms")
    return derive_features(era)[["station", "hour"] + ERA5_FEATURES]


def add_era5(df, era):
    """Attach ERA5-Land values from the nearest full hour of each sample."""
    df = df.assign(hour=df["valid"].dt.round("h"))
    return df.merge(era, on=["station", "hour"], how="left").drop(columns="hour")


def fetch_point(cfg, lon, lat, t):
    """ERA5-Land fog features for one point and the nearest full hour (one small GEE request).
    Raises LookupError if the hour is not in the archive yet."""
    hour = pd.Timestamp(t).round("h")
    raw = _region(cfg, _collection(cfg), lon, lat, hour.isoformat(), (hour + pd.Timedelta("1h")).isoformat())
    if raw.empty or raw[cfg["era5"]["bands"]].isna().any(axis=None):
        raise LookupError(f"no ERA5-Land data for {hour} yet")
    return derive_features(raw).iloc[0][ERA5_FEATURES].astype(float).to_dict()
