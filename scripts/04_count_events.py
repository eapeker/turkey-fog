import sys
sys.path.append("src")
from metar import load_metar, mark_fog
import pandas as pd

df = load_metar("data/raw/LTAC_2018_2024.csv")
df = mark_fog(df)
print("sisli rapor sayısı: ", df["is_fog"].sum())

print("-" * 80)

fog = df[df["is_fog"]].sort_values("valid")
gap = fog["valid"].diff()
fog["event_id"] = (gap > pd.Timedelta("1h")).cumsum()
print("sis olayı sayısı: ", fog["event_id"].nunique())

print("-" * 80)

events = fog.groupby("event_id")["valid"].agg(["min", "max", "count"])
events["duration"] = events["max"] - events["min"]
print(events.head(10))
print(events["duration"].describe())
print(events["duration"].max(), events["duration"].min())

uzun = events["duration"].idxmax()
print(fog[fog["event_id"] == uzun][["valid", "vsby", "wxcodes"]])