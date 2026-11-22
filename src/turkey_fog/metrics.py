import numpy as np
import pandas as pd


def scores(y, pred):
    """POD (share of fog cases found), FAR (share of fog alarms that were wrong)
    and CSI (hits / (hits + misses + false alarms)) for the fog class."""
    y, pred = np.asarray(y), np.asarray(pred)
    h = int(((pred == 1) & (y == 1)).sum())
    m = int(((pred == 0) & (y == 1)).sum())
    f = int(((pred == 1) & (y == 0)).sum())
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else 0.0
    csi = h / (h + m + f) if h + m + f else np.nan
    return pod, far, csi


def bootstrap_csi(test, preds, n=1000, seed=0):
    """CSI of each model on n resampled test sets, resampling whole nights
    (rows of the same night stay together)."""
    rng = np.random.default_rng(seed)
    t = test.assign(**preds)
    groups = [g for _, g in t.groupby("night_id")]
    boot = {k: [] for k in preds}
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k in preds:
            boot[k].append(scores(s["label"], s[k])[2])
    return {k: np.array(v) for k, v in boot.items()}
