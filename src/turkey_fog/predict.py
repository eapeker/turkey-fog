import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from turkey_fog import era5, seviri
from turkey_fog.features import window_features


def load_model(name, model_dir="models"):
    """A saved model with its feature list and probability threshold."""
    return joblib.load(Path(model_dir) / f"{name}.joblib")


def apply_model(m, features):
    """Fog probability and yes/no decision for one sample (a dict of feature values)."""
    x = pd.DataFrame([features])[m["features"]]
    p = float(m["model"].predict_proba(x)[0, 1])
    return p, p > m["threshold"]


def is_night(cfg, lon, lat, t):
    from pyorbital.astronomy import sun_zenith_angle
    sza = float(sun_zenith_angle(pd.Timestamp(t).to_pydatetime(), lon, lat))
    return sza > cfg["sampling"]["night_sza"], sza


def _cache_file(cfg, kind, lon, lat, t):
    folder = Path(cfg["predict"]["cache"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{kind}_{lon:.3f}_{lat:.3f}_{pd.Timestamp(t):%Y%m%d_%H%M}.json"


def seviri_features_at(cfg, lon, lat, t):
    """12 SEVIRI features for one point and time (cached after the first download)."""
    f = _cache_file(cfg, "seviri", lon, lat, t)
    if f.exists():
        return json.loads(f.read_text())
    sv = cfg["seviri"]
    coll = seviri.open_collection(cfg)
    p = seviri.nearest_product(coll, pd.Timestamp(t).to_pydatetime(), sv["search_minutes"])
    if p is None:
        raise LookupError(f"no SEVIRI scan within ±{sv['search_minutes']} min of {t}")
    win = seviri.extract_window(p, lon, lat, sv["channels"], sv["window"], cfg["paths"]["tmp"])
    if (win < sv["min_bt"]).any():
        raise ValueError(f"invalid brightness temperatures (below {sv['min_bt']} K) in the window")
    out = {k: float(v) for k, v in window_features(win, sv["channels"]).items()}
    out["product"] = str(p)
    f.write_text(json.dumps(out))
    return out


def era5_features_at(cfg, lon, lat, t, timeout):
    """ERA5-Land features, or (None, reason) when they cannot be used."""
    lag = cfg["predict"]["era5_lag_days"]
    if pd.Timestamp(t) > pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=lag):
        return None, (f"ERA5 is not available for {t} yet. "
                      f"ERA5 data is published about {lag} days after real time.")
    f = _cache_file(cfg, "era5", lon, lat, t)
    if f.exists():
        return json.loads(f.read_text()), None
    pool = ThreadPoolExecutor(max_workers=1)
    job = pool.submit(era5.fetch_point, cfg, lon, lat, t)
    try:
        values = job.result(timeout=timeout)
    except TimeoutError:
        return None, f"ERA5 download took longer than {timeout} s."
    except Exception as e:
        return None, f"ERA5 could not be read ({e})."
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    f.write_text(json.dumps(values))
    return values, None


def predict_point(cfg, lon, lat, t, use_era5=True, timeout=None, model_dir="models"):
    """Fog prediction for one point and time. Returns a dict with the decision and
    notes explaining which model was used and why."""
    t = pd.Timestamp(t)
    name = cfg["predict"]["model"]
    timeout = timeout or cfg["predict"]["era5_timeout"]
    night, sza = is_night(cfg, lon, lat, t)
    if not night:
        raise ValueError(f"the sun is up (solar zenith angle {sza:.0f}°); "
                         "the model was trained on night-time data only")
    feats = seviri_features_at(cfg, lon, lat, t)
    notes = []
    era = None
    if use_era5:
        era, reason = era5_features_at(cfg, lon, lat, t, timeout)
        if reason:
            notes.append(reason)
    else:
        notes.append("ERA5 switched off by the user (--no-era5).")
    if era:
        feats.update(era)
        name += "_era5"
    else:
        notes.append("Using the SEVIRI-only model.")
    p, fog = apply_model(load_model(name, model_dir), feats)
    return {"time": str(t), "lon": lon, "lat": lat, "product": feats.get("product"),
            "model": name, "probability": p, "threshold": load_model(name, model_dir)["threshold"],
            "fog": fog, "notes": notes}


def predict_example(path, model_dir="models"):
    """Offline demo: a saved test-set night with its SEVIRI and ERA5 features.
    Runs the SEVIRI-only and the SEVIRI + ERA5 model and shows the METAR truth."""
    ex = json.loads(Path(path).read_text())
    out = []
    for name in [ex["model"], ex["model"] + "_era5"]:
        m = load_model(name, model_dir)
        p, fog = apply_model(m, ex["features"])
        out.append({"model": name, "probability": p, "threshold": m["threshold"], "fog": fog})
    return ex, out


def make_examples(cfg, test, model_dir="models", folder="examples"):
    """Pick two test-set nights the SEVIRI + ERA5 model gets right, one fog and one clear,
    each with a typical (median) probability among the correct cases, and save them."""
    name = cfg["predict"]["model"]
    m = load_model(name + "_era5", model_dir)
    t = test.copy()
    t["p"] = m["model"].predict_proba(t[m["features"]])[:, 1]
    t["pred"] = (t["p"] > m["threshold"]).astype(int)
    Path(folder).mkdir(exist_ok=True)
    cols = sorted(set(load_model(name, model_dir)["features"]) | set(m["features"]))
    saved = []
    for kind, label in [("fog", 1), ("clear", 0)]:
        ok = t[(t["label"] == label) & (t["pred"] == label)]
        r = ok.iloc[(ok["p"] - ok["p"].median()).abs().argsort().iloc[0]]
        ex = {"kind": kind, "station": r["station"], "time": str(r["valid"]),
              "lon": cfg["stations"][r["station"]]["lon"], "lat": cfg["stations"][r["station"]]["lat"],
              "metar_label": int(r["label"]), "model": name,
              "note": "Test-set night (2023-2024, not seen in training), picked as a typical correctly "
                      "classified case for this demo. Overall skill is in reports/results.csv.",
              "features": {c: float(r[c]) for c in cols}}
        f = Path(folder) / f"{kind}.json"
        f.write_text(json.dumps(ex, indent=2))
        saved.append(f)
    return saved
