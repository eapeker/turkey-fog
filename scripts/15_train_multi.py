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
TRAIN_INFO = "LTAN, LTBY, LTCE 2018-2022"

# --- data: train on three stations (2018-2022), test on unseen LTAC (2023-2024) ---
multi = pd.read_csv("data/processed/features_multi.csv", parse_dates=["valid"])
ltac = pd.read_csv("data/processed/features.csv", parse_dates=["valid"]).assign(station="LTAC")
for d in (multi, ltac):
    d["satellite"] = d["product"].str[:4]
    d["night_id"] = d["station"] + "_" + (d["valid"] - pd.Timedelta(hours=12)).dt.date.astype(str)
train = multi[multi["valid"].dt.year <= 2022]
test = ltac[ltac["valid"].dt.year >= 2023]
print(train.groupby("station")["label"].value_counts().unstack())
print("test LTAC:", len(test), "rows,", int(test["label"].sum()), "fog")

BANDS = [c for c in multi.columns if c.startswith(("IR_", "WV_"))]
FEATURES = BANDS + ["btd_center", "btd_mean", "btd_max", "btd_std"]

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
        print(f"{name:7s} {s:4s} n={len(g):3d} POD={pod:.2f} FAR={far:.2f} CSI={csi:.2f}")

# --- train with night-grouped CV threshold, evaluate once on test ---
runs = {}

def fit_eval(name, model):
    oof = cross_val_predict(model, train[FEATURES], train["label"], groups=train["night_id"],
                            cv=GroupKFold(5), method="predict_proba")[:, 1]
    ths = np.arange(0.05, 0.96, 0.05)
    best = ths[np.nanargmax([scores(train["label"], (oof > t).astype(int))[2] for t in ths])]
    for s, g in train.assign(oof=oof).groupby("station"):
        print(f"  {name} CV {s}: CSI={scores(g['label'], (g['oof'] > best).astype(int))[2]:.2f}")
    model.fit(train[FEATURES], train["label"])
    pred = (model.predict_proba(test[FEATURES])[:, 1] > best).astype(int)
    print(f"{name}: probability threshold {best:.2f}")
    report(name, test, pred)
    pod, far, csi = scores(test["label"], pred)
    with mlflow.start_run(run_name=name) as run:
        runs[name] = run.info.run_id
        mlflow.log_params({"estimator": str(model), "threshold": round(float(best), 2), "train": TRAIN_INFO})
        mlflow.log_metrics({"pod": pod, "far": far, "csi": csi})
    return pred

# --- threshold baseline, threshold chosen on the new training data ---
ks = np.arange(0.0, 8.01, 0.25)
best_k = ks[np.nanargmax([scores(train["label"], (train["btd_center"] > k).astype(int))[2] for k in ks])]
print(f"thresh: BTD threshold {best_k:.2f} K (chosen on train)")
preds = {"thresh": (test["btd_center"] > best_k).astype(int).values}
report("thresh", test, preds["thresh"])
with mlflow.start_run(run_name="thresh") as run:
    runs["thresh"] = run.info.run_id
    mlflow.log_params({"estimator": f"btd_center > {best_k:.2f} K", "train": TRAIN_INFO})
    pod, far, csi = scores(test["label"], preds["thresh"])
    mlflow.log_metrics({"pod": pod, "far": far, "csi": csi})

preds["logreg"] = fit_eval("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
preds["xgb"] = fit_eval("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8))
preds["rf"] = fit_eval("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=0))
preds["mlp"] = fit_eval("mlp", make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=2000, random_state=0)))

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
    print(f"{k:7s} CSI 95% interval: [{lo:.2f} {hi:.2f}]")
    with mlflow.start_run(run_id=runs[k]):
        mlflow.log_metrics({"csi_lo": lo, "csi_hi": hi})

for k in preds:
    if k != "xgb":
        d = np.array(boot["xgb"]) - np.array(boot[k])
        print(f"xgb - {k:6s} CSI 95% interval:", np.nanpercentile(d, [2.5, 97.5]).round(2))