import sys
import numpy as np
import pandas as pd
sys.path.append("src")
from metar import load_metar, mark_fog
from pyorbital.astronomy import sun_zenith_angle

STATIONS = ["LTCE", "LTAN", "LTBY"]
N_EVENTS = 75
GAP = {"LTCE": "1h", "LTAN": "2h", "LTBY": "2h"}   # 2h for hourly stations

raw = load_metar("data/raw/stations_2018_2024.csv")


def pick(g, k=6):
    idx = np.linspace(0, len(g) - 1, min(len(g), k)).round().astype(int)
    return g.iloc[idx]


parts = []
for st in STATIONS:
    df = mark_fog(raw[raw["station"] == st].sort_values("valid").copy())
    lon, lat = df["lon"].iloc[0], df["lat"].iloc[0]
    df["sza"] = sun_zenith_angle(df["valid"].values, lon, lat)

    fog = df[df["is_fog"]].copy()
    fog["event"] = (fog["valid"].diff() > pd.Timedelta(GAP[st])).cumsum()
    fog_night = fog[fog["sza"] > 95]
    events = pd.Series(fog_night["event"].unique()).sample(
        n=min(N_EVENTS, fog_night["event"].nunique()), random_state=42)
    fog_night = fog_night[fog_night["event"].isin(events)]
    pos = fog_night.loc[fog_night.groupby("event", group_keys=False).apply(pick).index]

    cand = df[(df["sza"] > 95) & (~df["is_fog"]) & df["vsby"].notna()]
    fog_times = fog[["valid"]].rename(columns={"valid": "fog_time"})
    cand = pd.merge_asof(cand, fog_times, left_on="valid", right_on="fog_time", direction="nearest")
    cand = cand[(cand["valid"] - cand["fog_time"]).abs() > pd.Timedelta("3h")]
    neg = cand.sample(n=3 * len(pos), random_state=42)

    s = pd.concat([pos.assign(label=1), neg.assign(label=0)])
    s = s.assign(station=st, lon=lon, lat=lat)
    print(st, "events:", fog_night["event"].nunique(), "positives:", len(pos), "negatives:", len(neg))
    parts.append(s[["station", "lon", "lat", "valid", "label", "event", "vsby", "wxcodes", "sza"]])

sample = pd.concat(parts).sort_values(["station", "valid"])
sample.to_csv("data/processed/sample_times_multi.csv", index=False)
print(sample.groupby("station")["label"].value_counts().unstack())