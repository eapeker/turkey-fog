import pandas as pd
import numpy as np

df = pd.read_csv("data/processed/features.csv", parse_dates=["valid"])
train = df[df["valid"].dt.year <= 2022]
test = df[df["valid"].dt.year >= 2023]
print("train:", len(train), int(train["label"].sum()))
print("test: ", len(test), int(test["label"].sum()))

def scores(y, pred):
    h = int(((pred == 1) & (y == 1)).sum())
    m = int(((pred == 0) & (y == 1)).sum())
    f = int(((pred == 1) & (y == 0)).sum())
    return h / (h + m), f / (h + f) if h + f else 0.0, h / (h + m + f)

thresholds = np.arange(0, 8.01, 0.1)
csi = [scores(train["label"], (train["btd_center"] > t).astype(int))[2] for t in thresholds]
best = thresholds[int(np.argmax(csi))]
print(f"best threshold: {best:.1f} K, train CSI: {max(csi):.2f}")

pod, far, c = scores(test["label"], (test["btd_center"] > best).astype(int))
print(f"test POD={pod:.2f} FAR={far:.2f} CSI={c:.2f}")

for name, d in [("train", train), ("test", test)]:
    fog = d[d["label"] == 1]
    print(name, "fog BTD median:", round(fog["btd_center"].median(), 2),
          "| below 0 K:", int((fog["btd_center"] < 0).sum()),
          "|", d["product"].str[:4].value_counts().to_dict())


print(df.groupby([df["product"].str[:4], "label"])["btd_center"].median().round(2))