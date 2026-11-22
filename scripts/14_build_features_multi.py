from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = ["IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"]

rows = []
for f in sorted(Path("data/processed/windows_multi").glob("*.npz")):
    d = np.load(f)
    w = d["win"]
    if (w < 150).any():
        print(f.name, "skipped: brightness temperature below 150 K")
        continue    
    btd = w[5] - w[0]
    row = {"station": str(d["station"]), "valid": str(d["valid"]), "label": int(d["label"]), "product": str(d["product"]),
           "btd_center": btd[2, 2], "btd_mean": btd.mean(), "btd_max": btd.max(), "btd_std": btd.std()}
    row.update({f"{c}_center": w[i, 2, 2] for i, c in enumerate(CHANNELS)})
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/processed/features_multi.csv", index=False)
print(df.shape, int(df.isna().sum().sum()))
print(df.groupby("station")["label"].value_counts().unstack())
print(df.groupby("label")[["btd_center", "btd_max"]].describe().round(2).T)

df = pd.read_csv("data/processed/features_multi.csv")
print(df[df["btd_max"] > 30][["station", "valid", "label", "btd_center", "btd_max", "IR_039_center", "IR_108_center"]])


d = np.load("data/processed/windows_multi/LTCE_20210413_0039.npz")
print(d["win"][0].round(1))

ltac = pd.read_csv("data/processed/features.csv")
print(ltac["btd_max"].max(), ltac["btd_center"].min())