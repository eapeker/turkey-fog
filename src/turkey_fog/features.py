from pathlib import Path

import numpy as np
import pandas as pd

from turkey_fog.seviri import window_path


def window_features(win, channels):
    """12 features from one window (channel, row, col): the centre value of each channel
    and four statistics of the IR10.8 − IR3.9 brightness temperature difference (BTD)."""
    c = win.shape[1] // 2
    btd = win[channels.index("IR_108")] - win[channels.index("IR_039")]
    row = {"btd_center": btd[c, c], "btd_mean": btd.mean(), "btd_max": btd.max(), "btd_std": btd.std()}
    row.update({f"{ch}_center": win[i, c, c] for i, ch in enumerate(channels)})
    return row


def build_features(cfg, sample):
    """One row per sample whose window exists. Windows with any brightness temperature
    below seviri.min_bt (invalid values) are skipped."""
    sv = cfg["seviri"]
    rows = []
    for r in sample.itertuples():
        f = window_path(cfg["paths"]["windows"], r.station, r.valid)
        if not f.exists():
            continue
        d = np.load(f)
        w = d["win"]
        if (w < sv["min_bt"]).any():
            print(f.name, f"skipped: brightness temperature below {sv['min_bt']} K")
            continue
        row = {"station": r.station, "valid": r.valid, "label": r.label, "product": str(d["product"])}
        row.update(window_features(w, sv["channels"]))
        rows.append(row)
    return pd.DataFrame(rows)
