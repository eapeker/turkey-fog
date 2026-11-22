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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("turkey-fog-multi")
TRAIN_INFO = "LTAC, LTAN, LTBY, LTCE 2018-2022"

# --- data: train on all four stations (2018-2022), test on LTAC (2023-2024) ---
multi = pd.read_csv("data/processed/features_multi_era5.csv", parse_dates=["valid"])
ltac = pd.read_csv("data/processed/features_era5.csv", parse_dates=["valid"])
for d in (multi, ltac):
    d["satellite"] = d["product"].str[:4]
    d["night_id"] = d["station"] + "_" + (d["valid"] - pd.Timedelta(hours=12)).dt.date.astype(str)
train = pd.concat([multi, ltac])
train = train[train["valid"].dt.year <= 2022]
test = ltac[ltac["valid"].dt.year >= 2023]
print(train.groupby("station")["label"].value_counts().unstack())
print("test LTAC:", len(test), "rows,", int(test["label"].sum()), "fog")

BANDS = [c for c in multi.columns if c.startswith(("IR_", "WV_"))]
FEATURES = BANDS + ["btd_center", "btd_mean", "btd_max", "btd_std"]
ERA5 = ["temperature_2m", "dpd", "t2m_minus_skin", "wind10"]

def feature_set(feats):
    if feats == FEATURES:
        return "SEVIRI"
    if feats == ERA5:
        return "ERA5"
    return "SEVIRI+ERA5"

# --- metrics ---
def scores(y, pred):
    h = int(((pred == 1) & (y == 1)).sum())
    m = int(((pred == 0) & (y == 1)).sum())
    f = int(((pred == 1) & (y == 0)).sum())
    pod = h / (h + m) if h + m else np.nan
    far = f / (h + f) if h + f else 0.0
    csi = h / (h + m + f) if h + m + f else np.nan
    return pod, far, csi

def report(name, d, pred):
    d = d.assign(pred=pred)
    for s, g in [("all", d)] + list(d.groupby("satellite")):
        pod, far, csi = scores(g["label"], g["pred"])
        print(f"{name:15s} {s:4s} n={len(g):3d} POD={pod:.2f} FAR={far:.2f} CSI={csi:.2f}")

# --- train with night-grouped CV threshold, evaluate once on test ---
runs = {}

def fit_eval(name, model, feats):
    oof = cross_val_predict(model, train[feats], train["label"], groups=train["night_id"],
                            cv=GroupKFold(5), method="predict_proba")[:, 1]
    ths = np.arange(0.05, 0.96, 0.05)
    best = ths[np.nanargmax([scores(train["label"], (oof > t).astype(int))[2] for t in ths])]
    for s, g in train.assign(oof=oof).groupby("station"):
        print(f"  {name} CV {s}: CSI={scores(g['label'], (g['oof'] > best).astype(int))[2]:.2f}")
    model.fit(train[feats], train["label"])
    pred = (model.predict_proba(test[feats])[:, 1] > best).astype(int)
    print(f"{name}: probability threshold {best:.2f}")
    report(name, test, pred)
    pod, far, csi = scores(test["label"], pred)
    with mlflow.start_run(run_name=name) as run:
        runs[name] = run.info.run_id
        mlflow.log_params({"estimator": str(model), "threshold": round(float(best), 2),
                           "train": TRAIN_INFO, "features": feature_set(feats)})
        mlflow.log_metrics({"pod": pod, "far": far, "csi": csi})
    return pred

# --- threshold baseline, threshold chosen on the training data ---
ks = np.arange(0.0, 8.01, 0.25)
best_k = ks[np.nanargmax([scores(train["label"], (train["btd_center"] > k).astype(int))[2] for k in ks])]
print(f"thresh: BTD threshold {best_k:.2f} K (chosen on train)")
preds = {"thresh": (test["btd_center"] > best_k).astype(int).values}
report("thresh", test, preds["thresh"])
with mlflow.start_run(run_name="thresh") as run:
    runs["thresh"] = run.info.run_id
    mlflow.log_params({"estimator": f"btd_center > {best_k:.2f} K", "train": TRAIN_INFO, "features": "SEVIRI"})
    pod, far, csi = scores(test["label"], preds["thresh"])
    mlflow.log_metrics({"pod": pod, "far": far, "csi": csi})

# --- the same four models with three feature sets: SEVIRI, SEVIRI+ERA5, ERA5 only ---
def models():
    return {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "xgb": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
            "rf": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0),
            "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=2000, random_state=0))}

for k, m in models().items():
    preds[k] = fit_eval(k, m, FEATURES)
for k, m in models().items():
    preds[k + "_era5"] = fit_eval(k + "_era5", m, FEATURES + ERA5)
for k, m in models().items():
    preds[k + "_era5only"] = fit_eval(k + "_era5only", m, ERA5)

# --- bootstrap 95% intervals, resampling whole nights ---
rng = np.random.default_rng(0)
t = test.assign(**preds)
groups = [g for _, g in t.groupby("night_id")]
boot = {k: [] for k in preds}
for _ in range(1000):
    s = pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))])
    for k in preds:
        boot[k].append(scores(s["label"], s[k])[2])

for k, v in boot.items():
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    print(f"{k:15s} CSI 95% interval: [{lo:.2f} {hi:.2f}]")
    with mlflow.start_run(run_id=runs[k]):
        mlflow.log_metrics({"csi_lo": lo, "csi_hi": hi})

# --- does ERA5 help the satellite model? (SEVIRI+ERA5 minus SEVIRI) ---
for k in ["logreg", "xgb", "rf", "mlp"]:
    d = np.array(boot[k + "_era5"]) - np.array(boot[k])
    print(f"{k + '_era5':15s} - {k:8s} CSI 95% interval:", np.nanpercentile(d, [2.5, 97.5]).round(2))

# --- does the satellite help on top of ERA5? (SEVIRI+ERA5 minus ERA5 only) ---
for k in ["logreg", "xgb", "rf", "mlp"]:
    d = np.array(boot[k + "_era5"]) - np.array(boot[k + "_era5only"])
    print(f"{k + '_era5':15s} - {k + '_era5only':15s} CSI 95% interval:", np.nanpercentile(d, [2.5, 97.5]).round(2))