import sys
import pandas as pd
import numpy as np
sys.path.append("src")
from metar import load_metar, mark_fog
from pyorbital.astronomy import sun_zenith_angle

df = mark_fog(load_metar("data/raw/LTAC_2018_2024.csv"))
df["sza"] = sun_zenith_angle(df["valid"].values, 32.9951, 40.1281)
night = df[df["sza"] > 95]
print(len(df), len(night), int(night["is_fog"].sum()))

fog = df[df["is_fog"]].copy()
fog["event"] = (fog["valid"].diff() > pd.Timedelta("1h")).cumsum()
fog_night = fog[fog["sza"] > 95]
print(fog["event"].nunique(), fog_night["event"].nunique())
print(fog_night.groupby("event").size().describe())

def pick(g, k=6):
    idx = np.linspace(0, len(g) - 1, min(len(g), k)).round().astype(int)
    return g.iloc[idx]

pos = fog_night.loc[fog_night.groupby("event", group_keys=False).apply(pick).index]
print("positives:", len(pos))

cand = df[(df["sza"] > 95) & (~df["is_fog"]) & df["vsby"].notna()].sort_values("valid")
fog_times = fog[["valid"]].rename(columns={"valid": "fog_time"}).sort_values("fog_time")
cand = pd.merge_asof(cand, fog_times, left_on="valid", right_on="fog_time", direction="nearest")
cand = cand[(cand["valid"] - cand["fog_time"]).abs() > pd.Timedelta("3h")]
neg = cand.sample(n=3 * len(pos), random_state=42)
print("candidates:", len(cand), "negatives:", len(neg))

sample = pd.concat([pos.assign(label=1), neg.assign(label=0)])
sample = sample[["valid", "label", "event", "vsby", "wxcodes", "sza"]].sort_values("valid")
sample.to_csv("data/processed/sample_times.csv", index=False)
print(sample["label"].value_counts())