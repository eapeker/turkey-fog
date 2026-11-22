from pathlib import Path
import ee
import pandas as pd

ee.Initialize(project="ee-eminatabeypeker")

BANDS = ["temperature_2m", "dewpoint_temperature_2m", "skin_temperature",
         "u_component_of_wind_10m", "v_component_of_wind_10m"]

stations = {"LTAC": (32.9951, 40.1281)}
multi = pd.read_csv("data/processed/sample_times_multi.csv")
for st, g in multi.groupby("station"):
    stations[st] = (g["lon"].iloc[0], g["lat"].iloc[0])
print(stations)

out = Path("data/raw/era5land")
out.mkdir(parents=True, exist_ok=True)
coll = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY").select(BANDS)

for st, (lon, lat) in stations.items():
    point = ee.Geometry.Point(lon, lat)
    for year in range(2018, 2025):
        f = out / f"{st}_{year}.csv"
        if f.exists():
            continue
        parts = []
        for m in range(1, 13):
            start = pd.Timestamp(year, m, 1)
            end = start + pd.offsets.MonthBegin(1)
            rows = coll.filterDate(str(start.date()), str(end.date())).getRegion(point, 11132).getInfo()
            parts.append(pd.DataFrame(rows[1:], columns=rows[0]))
        df = pd.concat(parts)
        df.to_csv(f, index=False)
        print(st, year, len(df), flush=True)