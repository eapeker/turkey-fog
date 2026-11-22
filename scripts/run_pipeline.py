"""Run the turkey-fog pipeline step by step, e.g.  python scripts/run_pipeline.py select"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from turkey_fog.config import load_config
from turkey_fog.era5 import add_era5, download_era5land, load_era5land
from turkey_fog.features import build_features
from turkey_fog.metar import select_times
from turkey_fog.seviri import download_windows
from turkey_fog.predict import make_examples
from turkey_fog.train import run_experiments, split


def load_sample(cfg):
    return pd.read_csv(cfg["paths"]["sample_times"], parse_dates=["valid"])


def step_select(cfg, args):
    sample = select_times(cfg)
    out = Path(cfg["paths"]["sample_times"])
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(sample.groupby("station")["label"].value_counts().unstack())
    print("saved", out)


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
    results, diffs = run_experiments(cfg, df, config_path=args.config)
    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/results.csv", index=False)
    diffs.to_csv("reports/comparisons.csv", index=False)
    pd.set_option("display.width", 200)
    print(results[["name", "features", "threshold", "pod", "far", "csi", "csi_lo", "csi_hi"]].round(2).to_string(index=False))
    print(diffs.round(2).to_string(index=False))
    print("saved reports/results.csv, reports/comparisons.csv and models/")


def step_examples(cfg, args):
    df = pd.read_csv(cfg["paths"]["features"], parse_dates=["valid"])
    _, test = split(cfg, df)
    for f in make_examples(cfg, test):
        print("saved", f)


STEPS = {"select": step_select, "download": step_download, "era5": step_era5,
         "features": step_features, "train": step_train, "examples": step_examples}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n", type=int, default=None, help="download only the first n missing windows (for testing)")
    args = parser.parse_args()
    STEPS[args.step](load_config(args.config), args)
