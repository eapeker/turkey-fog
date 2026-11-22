import gc
import shutil
import warnings
import zipfile
from datetime import timedelta
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)


def open_collection(cfg):
    """Log in to the EUMETSAT Data Store with ~/.eumdac/credentials (key,secret)."""
    import eumdac
    key, secret = (Path.home() / ".eumdac" / "credentials").read_text().strip().split(",")
    token = eumdac.AccessToken((key, secret))
    return eumdac.DataStore(token).get_collection(cfg["seviri"]["collection"])


def nearest_product(coll, t, minutes):
    """The SEVIRI scan whose sensing end is closest to t, within ± minutes."""
    prods = list(coll.search(dtstart=t - timedelta(minutes=minutes), dtend=t + timedelta(minutes=minutes)))
    return min(prods, key=lambda p: abs(p.sensing_end - t)) if prods else None


def extract_window(p, lon, lat, channels, size, tmp):
    """Download one scan, cut a size × size window of brightness temperatures around (lon, lat),
    delete the raw files and return an array (channel, row, col)."""
    from satpy import Scene
    tmp = Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    zip_path = tmp / f"{p}.zip"
    with p.open() as src, open(zip_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    with zipfile.ZipFile(zip_path) as z:
        nat = [n for n in z.namelist() if n.endswith(".nat")][0]
        z.extract(nat, tmp)
    scn = Scene(reader="seviri_l1b_native", filenames=[str(tmp / nat)])
    scn.load(channels)
    local = scn.crop(ll_bbox=(lon - 1, lat - 1, lon + 1, lat + 1))
    col, row = local["IR_108"].attrs["area"].get_array_indices_from_lonlat(lon, lat)
    h = size // 2
    win = np.stack([local[c].values[row - h:row + h + 1, col - h:col + h + 1] for c in channels])
    del scn, local
    gc.collect()
    zip_path.unlink()
    (tmp / nat).unlink()
    return win


def window_path(folder, station, valid):
    return Path(folder) / f"{station}_{valid:%Y%m%d_%H%M}.npz"


def download_windows(cfg, sample, n=None):
    """Download a window for every row of the sample table. Existing files are skipped,
    so the step can be stopped and restarted."""
    sv = cfg["seviri"]
    out = Path(cfg["paths"]["windows"])
    out.mkdir(parents=True, exist_ok=True)
    todo = sample[[not window_path(out, r.station, r.valid).exists() for r in sample.itertuples()]]
    print(len(sample) - len(todo), "windows already there,", len(todo), "missing")
    if n:
        todo = todo.head(n)
    if todo.empty:
        return
    coll = open_collection(cfg)
    for i, r in todo.iterrows():
        try:
            p = nearest_product(coll, r.valid.to_pydatetime(), sv["search_minutes"])
            if p is None:
                print(i, r.station, r.valid, "NO SCAN", flush=True)
                continue
            win = extract_window(p, r.lon, r.lat, sv["channels"], sv["window"], cfg["paths"]["tmp"])
            np.savez(window_path(out, r.station, r.valid), win=win, label=r.label,
                     valid=str(r.valid), product=str(p), station=r.station)
            print(i, r.station, r.valid, "ok", flush=True)
        except Exception as e:
            print(i, r.station, r.valid, "ERROR", e, flush=True)