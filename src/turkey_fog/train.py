from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from turkey_fog.era5 import ERA5_FEATURES


def seviri_features(cfg):
    """The 12 satellite features: 8 channel centre values + 4 BTD statistics."""
    return [f"{c}_center" for c in cfg["seviri"]["channels"]] + ["btd_center", "btd_mean", "btd_max", "btd_std"]


def feature_sets(cfg):
    """Feature set name -> (columns, suffix added to the model name)."""
    sev = seviri_features(cfg)
    return {"SEVIRI": (sev, ""), "SEVIRI+ERA5": (sev + ERA5_FEATURES, "_era5"), "ERA5": (ERA5_FEATURES, "_era5only")}


def make_models():
    """Fresh, untrained copies of the four models."""
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "xgb": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
            "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
            "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3,
                                                                 max_iter=2000, random_state=0))}


def scores(y, pred):
    """POD (hit rate), FAR (share of false alarms) and CSI for the fog class."""
    h = int(((pred == 1) & (y == 1)).sum())
    m = int(((pred == 0) & (y == 1)).sum())
    f = int(((pred == 1) & (y == 0)).sum())
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else 0.0
    csi = h / (h + m + f) if h + m + f else np.nan
    return pod, far, csi


def split(cfg, df):
    """Train: all stations in the training years. Test: the test station in the test years."""
    s = cfg["split"]
    df = df.assign(satellite=df["product"].str[:4],
                   night_id=df["station"] + "_" + (df["valid"] - pd.Timedelta(hours=12)).dt.date.astype(str))
    year = df["valid"].dt.year
    train = df[year.between(*s["train_years"])]
    test = df[(df["station"] == s["test_station"]) & year.between(*s["test_years"])]
    return train, test


def fit_with_threshold(model, train, feats):
    """Pick the probability threshold with the best CSI on night-grouped 5-fold CV predictions,
    then fit on all training data."""
    oof = cross_val_predict(model, train[feats], train["label"], groups=train["night_id"],
                            cv=GroupKFold(5), method="predict_proba")[:, 1]
    ths = np.arange(0.05, 0.96, 0.05)
    best = float(ths[np.nanargmax([scores(train["label"], (oof > t).astype(int))[2] for t in ths])])
    cv = {s: scores(g["label"], (g["oof"] > best).astype(int))[2]
          for s, g in train.assign(oof=oof).groupby("station")}
    model.fit(train[feats], train["label"])
    return model, best, cv


def bootstrap_csi(test, preds, n=1000, seed=0):
    """CSI of each model on 1000 resampled test sets, resampling whole nights."""
    rng = np.random.default_rng(seed)
    t = test.assign(**preds)
    groups = [g for _, g in t.groupby("night_id")]
    boot = {k: [] for k in preds}
    for _ in range(n):
        s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for k in preds:
            boot[k].append(scores(s["label"], s[k])[2])
    return {k: np.array(v) for k, v in boot.items()}


def run_experiments(cfg, df, model_dir="models"):
    """Threshold baseline + four models × three feature sets. Logs every run to MLflow,
    saves the fitted models and returns a results table."""
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    s = cfg["split"]
    train_info = f"all stations {s['train_years'][0]}-{s['train_years'][1]}"
    test_info = f"{s['test_station']} {s['test_years'][0]}-{s['test_years'][1]}"
    train, test = split(cfg, df)
    print(train.groupby("station")["label"].value_counts().unstack())
    print("test", test_info, len(test), "rows,", int(test["label"].sum()), "fog")
    Path(model_dir).mkdir(exist_ok=True)

    rows, preds = [], {}

    ks = np.arange(0.0, 8.01, 0.25)
    best_k = float(ks[np.nanargmax([scores(train["label"], (train["btd_center"] > k).astype(int))[2] for k in ks])])
    preds["thresh"] = (test["btd_center"] > best_k).astype(int).values
    rows.append({"name": "thresh", "model": "thresh", "features": "SEVIRI", "threshold": best_k})

    for fs_name, (feats, suffix) in feature_sets(cfg).items():
        for m_name, model in make_models().items():
            name = m_name + suffix
            model, th, cv = fit_with_threshold(model, train, feats)
            preds[name] = (model.predict_proba(test[feats])[:, 1] > th).astype(int)
            print(f"{name:16s} threshold {th:.2f}  CV CSI by station:",
                  " ".join(f"{k}={v:.2f}" for k, v in cv.items()))
            joblib.dump({"model": model, "features": feats, "threshold": th}, Path(model_dir) / f"{name}.joblib")
            rows.append({"name": name, "model": m_name, "features": fs_name, "threshold": th})

    boot = bootstrap_csi(test, preds)
    for r in rows:
        pod, far, csi = scores(test["label"], preds[r["name"]])
        lo, hi = np.nanpercentile(boot[r["name"]], [2.5, 97.5])
        r.update({"pod": pod, "far": far, "csi": csi, "csi_lo": lo, "csi_hi": hi})
        with mlflow.start_run(run_name=r["name"]):
            mlflow.log_params({"model": r["model"], "features": r["features"], "threshold": round(r["threshold"], 2),
                               "train": train_info, "test": test_info})
            mlflow.log_metrics({k: r[k] for k in ["pod", "far", "csi", "csi_lo", "csi_hi"]})
    results = pd.DataFrame(rows)

    diffs = []
    for m in make_models():
        for a, b in [(f"{m}_era5", m), (f"{m}_era5", f"{m}_era5only")]:
            lo, hi = np.nanpercentile(boot[a] - boot[b], [2.5, 97.5])
            diffs.append({"comparison": f"{a} - {b}", "csi_diff_lo": lo, "csi_diff_hi": hi})
    return results, pd.DataFrame(diffs)