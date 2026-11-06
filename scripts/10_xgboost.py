import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# --- data and split (same as 09_baseline.py) ---
df = pd.read_csv("data/processed/features.csv", parse_dates=["valid"])
df["satellite"] = df["product"].str[:4]
df["night_id"] = (df["valid"] - pd.Timedelta(hours=12)).dt.date
train = df[df["valid"].dt.year <= 2022]
test = df[df["valid"].dt.year >= 2023]

BANDS = [c for c in df.columns if c.startswith(("IR_", "WV_"))]
FEATURES = BANDS + ["btd_center", "btd_mean", "btd_max", "btd_std"]

# --- diagnosis: same-season non-fog BTD by satellite (no correction applied) ---
win = train[train["valid"].dt.month.isin([1, 2]) & (train["label"] == 0)]
print(win.groupby("satellite")["btd_center"].agg(["median", "count"]).round(2))

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
def fit_eval(name, model):
    oof = cross_val_predict(model, train[FEATURES], train["label"], groups=train["night_id"],
                            cv=GroupKFold(5), method="predict_proba")[:, 1]
    ths = np.arange(0.05, 0.96, 0.05)
    best = ths[np.nanargmax([scores(train["label"], (oof > t).astype(int))[2] for t in ths])]
    model.fit(train[FEATURES], train["label"])
    pred = (model.predict_proba(test[FEATURES])[:, 1] > best).astype(int)
    print(f"{name}: probability threshold {best:.2f}")
    report(name, test, pred)
    return pred

preds = {"thresh": (test["btd_center"] > 4.0).astype(int).values}
report("thresh", test, preds["thresh"])
preds["logreg"] = fit_eval("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)))
preds["xgb"] = fit_eval("xgb", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8))

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
    print(f"{k:7s} CSI 95% interval:", np.nanpercentile(v, [2.5, 97.5]).round(2))
d = np.array(boot["xgb"]) - np.array(boot["logreg"])
print("xgb - logreg CSI 95% interval:", np.nanpercentile(d, [2.5, 97.5]).round(2))
d2 = np.array(boot["xgb"]) - np.array(boot["thresh"])
print("xgb - thresh CSI 95% interval:", np.nanpercentile(d2, [2.5, 97.5]).round(2))