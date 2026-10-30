import gc
import shutil
import zipfile
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import eumdac
from satpy import Scene

warnings.filterwarnings("ignore", category=RuntimeWarning)

CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]
LON, LAT = 32.9951, 40.1281
N = None  

key, secret = (Path.home() / ".eumdac" / "credentials").read_text().strip().split(",")
token = eumdac.AccessToken((key, secret))
coll = eumdac.DataStore(token).get_collection("EO:EUM:DAT:MSG:HRSEVIRI")

tmp = Path("data/tmp")
out = Path("data/processed/windows")
tmp.mkdir(parents=True, exist_ok=True)
out.mkdir(parents=True, exist_ok=True)


def nearest_product(t):
    prods = list(coll.search(dtstart=t - timedelta(minutes=10), dtend=t + timedelta(minutes=10)))
    return min(prods, key=lambda p: abs(p.sensing_end - t)) if prods else None


def extract_window(p):
    zip_path = tmp / f"{p}.zip"
    with p.open() as src, open(zip_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with zipfile.ZipFile(zip_path) as z:
        nat = [n for n in z.namelist() if n.endswith(".nat")][0]
        z.extract(nat, tmp)
    scn = Scene(reader="seviri_l1b_native", filenames=[str(tmp / nat)])
    scn.load(CHANNELS)
    local = scn.crop(ll_bbox=(LON - 1, LAT - 1, LON + 1, LAT + 1))
    col, row = local["IR_108"].attrs["area"].get_array_indices_from_lonlat(LON, LAT)
    win = np.stack([local[c].values[row - 2:row + 3, col - 2:col + 3] for c in CHANNELS])
    del scn, local
    gc.collect()
    zip_path.unlink()
    (tmp / nat).unlink()
    return win


sample = pd.read_csv("data/processed/sample_times.csv", parse_dates=["valid"])
if N:
    sample = sample.head(N)

for i, r in sample.iterrows():
    f = out / f"{r.valid:%Y%m%d_%H%M}.npz"
    if f.exists():
        continue
    try:
        p = nearest_product(r.valid.to_pydatetime())
        if p is None:
            print(i, r.valid, "NO SCAN", flush=True)
            continue
        win = extract_window(p)
        np.savez(f, win=win, label=r.label, valid=str(r.valid), product=str(p))
        print(i, r.valid, "ok", flush=True)
    except Exception as e:
        print(i, r.valid, "ERROR", e, flush=True)