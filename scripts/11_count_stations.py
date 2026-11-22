import sys; sys.path.append("src")
from metar import load_metar, mark_fog
import pandas as pd

df = pd.concat([load_metar("data/raw/LTAC_2018_2024.csv"),
                load_metar("data/raw/stations_2018_2024.csv")])
df = mark_fog(df)

for st, g in df.groupby("station"):
    fog = g[g["is_fog"]].sort_values("valid")
    events = (fog["valid"].diff() > pd.Timedelta("1h")).cumsum()
    print(st, "rows:", len(g), "fog reports:", len(fog), "events:", events.nunique(),
          "test-year reports:", (fog["valid"].dt.year >= 2023).sum())