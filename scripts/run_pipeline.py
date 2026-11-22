"""Run the turkey-fog pipeline step by step, e.g.  python scripts/run_pipeline.py select"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from turkey_fog.config import load_config
from turkey_fog.era5 import add_era5, download_era5land, load_era5land
from turkey_fog.features import build_features
from turkey_fog.metar import select_times
from turkey_fog.seviri import download_windows, window_path
from turkey_fog.train import run_experiments


def load_sample(cfg):
    return pd.read_csv(cfg["paths"]["sample_times"], parse_dates=["valid"])


def step_select(cfg, args):
    sample = select_times(cfg)
    out = Path(cfg["paths"]["sample_times"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(sample.groupby("station")["label"].value_counts().unstack())
    print("saved", out)


def step_migrate(cfg, args):
    """One-off: copy windows downloaded by the old scripts into the new folder,
    so they are not downloaded again. Old LTAC files get a station field."""
    out = Path(cfg["paths"]["windows"])
    out.mkdir(parents=True, exist_ok=True)
    copied = 0
    for r in load_sample(cfg).itertuples():
        new = window_path(out, r.station, r.valid)
        if new.exists():
            continue
        if r.station == "LTAC":
            old = Path("data/processed/windows") / f"{r.valid:%Y%m%d_%H%M}.npz"
            if old.exists():
                d = dict(np.load(old))
                np.savez(new, **d, station="LTAC")
                copied += 1
        else:
            old = window_path("data/processed/windows_multi", r.station, r.valid)
            if old.exists():
                shutil.copy2(old, new)
                copied += 1
    print("copied", copied, "files to", out)


def step_download(cfg, args):
    download_windows(cfg, load_sample(cfg), n=args.n)


def step_era5(cfg, args):
    download_era5land(cfg)


def step_features(cfg, args):
    df = build_features(cfg, load_sample(cfg))
    df = add_era5(df, load_era5land(cfg))
    out = Path(cfg["paths"]["features"])
    df.to_csv(out, index=False)
    print(df.shape, "missing values:", int(df.isna().sum().sum()))
    print(df.groupby("station")["label"].value_counts().unstack())
    print("saved", out)


def step_train(cfg, args):
    df = pd.read_csv(cfg["paths"]["features"], parse_dates=["valid"])
    results, diffs = run_experiments(cfg, df)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/results.csv", index=False)
    diffs.to_csv("reports/comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(results[["name", "features", "threshold", "pod", "far", "csi", "csi_lo", "csi_hi"]].round(2).to_string(index=False))
    print(diffs.round(2).to_string(index=False))
    print("saved reports/results.csv, reports/comparisons.csv and models/")


STEPS = {"select": step_select, "migrate": step_migrate, "download": step_download,
         "era5": step_era5, "features": step_features, "train": step_train}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n", type=int, default=None, help="download only the first n missing windows (for testing)")
    args = parser.parse_args()
    STEPS[args.step](load_config(args.config), args)