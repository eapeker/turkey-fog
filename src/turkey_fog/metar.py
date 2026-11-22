import numpy as np
import pandas as pd
from pyorbital.astronomy import sun_zenith_angle


def load_metar(paths):
    """Read one or more IEM METAR CSV files ("M" = missing)."""
    if isinstance(paths, str):
        paths = [paths]
    df = pd.concat([pd.read_csv(p, na_values="M", low_memory=False) for p in paths])
    df["valid"] = pd.to_datetime(df["valid"])
    return df


def mark_fog(df):
    """Fog = visibility below 0.62 mi and FG or FZFG as a whole weather code."""
    words = df["wxcodes"].fillna("").str.split()
    df["is_fog"] = (df["vsby"] < 0.62) & words.apply(lambda w: "FG" in w or "FZFG" in w)
    return df


def mark_events(fog, gap):
    """Fog reports less than `gap` apart belong to the same event."""
    fog = fog.copy()
    fog["event"] = (fog["valid"].diff() > pd.Timedelta(gap)).cumsum()
    return fog


def pick_spread(g, k):
    """Pick up to k reports spread evenly over one event."""
    idx = np.linspace(0, len(g) - 1, min(len(g), k)).round().astype(int)
    return g.iloc[idx]


def select_station(df, lon, lat, gap, s):
    """Fog (label 1) and non-fog (label 0) night-time samples for one station.
    s = the "sampling" block of config.yaml."""
    df = mark_fog(df.sort_values("valid").copy())
    df["sza"] = sun_zenith_angle(df["valid"].values, lon, lat)
    night = df["sza"] > s["night_sza"]

    fog = mark_events(df[df["is_fog"]], gap)
    fog_night = fog[fog["sza"] > s["night_sza"]]
    events = pd.Series(fog_night["event"].unique()).sample(
        n=min(s["max_events"], fog_night["event"].nunique()), random_state=s["seed"])
    fog_night = fog_night[fog_night["event"].isin(events)]
    pos = fog_night.loc[fog_night.groupby("event", group_keys=False).apply(
        pick_spread, k=s["per_event"]).index]

    cand = df[night & ~df["is_fog"] & df["vsby"].notna()]
    fog_times = fog[["valid"]].rename(columns={"valid": "fog_time"})
    cand = pd.merge_asof(cand, fog_times, left_on="valid", right_on="fog_time", direction="nearest")
    cand = cand[(cand["valid"] - cand["fog_time"]).abs() > pd.Timedelta(s["neg_buffer"])]
    neg = cand.sample(n=s["neg_ratio"] * len(pos), random_state=s["seed"])

    return pd.concat([pos.assign(label=1), neg.assign(label=0)])


def select_times(cfg):
    """Run select_station for every station in config.yaml."""
    raw = load_metar(cfg["paths"]["metar"])
    y0, y1 = cfg["years"]
    raw = raw[raw["valid"].dt.year.between(y0, y1)]
    parts = []
    for st, p in cfg["stations"].items():
        s = select_station(raw[raw["station"] == st], p["lon"], p["lat"], p["gap"], cfg["sampling"])
        s = s.assign(station=st, lon=p["lon"], lat=p["lat"])
        parts.append(s[["station", "lon", "lat", "valid", "label", "event", "vsby", "wxcodes", "sza"]])
        print(st, "events:", s.loc[s["label"] == 1, "event"].nunique(),
              "fog:", int(s["label"].sum()), "non-fog:", int((s["label"] == 0).sum()))
    return pd.concat(parts).sort_values(["station", "valid"]).reset_index(drop=True)