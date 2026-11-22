"""Map of the four airports used in the project -> docs/figures/stations.png"""
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, "src")
from turkey_fog.config import load_config

NAMES = {"LTAC": "Ankara Esenboğa", "LTAN": "Konya", "LTBY": "Eskişehir", "LTCE": "Erzurum"}
# label position relative to the airport: (dx, dy) in degrees and text alignment
LABEL = {"LTAC": (0.35, 0.35, "left"), "LTAN": (0.35, -0.10, "left"),
         "LTBY": (-0.35, -0.55, "right"), "LTCE": (-0.35, 0.30, "right")}
TEST_COLOR, TRAIN_COLOR = "#c0392b", "#2c6fb7"

cfg = load_config()
test = cfg["split"]["test_station"]

counts = {}
f = Path(cfg["paths"]["features"])
if f.exists():
    df = pd.read_csv(f)
    counts = df.groupby("station")["label"].agg(fog="sum", total="count").to_dict("index")

fig = plt.figure(figsize=(10, 4.8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([25.5, 45.0, 35.5, 42.5], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="#f4f1ea")
ax.add_feature(cfeature.OCEAN, facecolor="#d6e6f2")
ax.add_feature(cfeature.LAKES, facecolor="#d6e6f2", edgecolor="#7a9cb8", linewidth=0.4)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#555555")
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=":", edgecolor="#555555")
gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#999999", alpha=0.6, linestyle="--")
gl.top_labels = gl.right_labels = False
gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": "#555555"}

halo = [pe.withStroke(linewidth=3, foreground="white")]
for st, p in cfg["stations"].items():
    is_test = st == test
    ax.plot(p["lon"], p["lat"], marker="*" if is_test else "o", markersize=16 if is_test else 9,
            color=TEST_COLOR if is_test else TRAIN_COLOR, markeredgecolor="black", markeredgewidth=0.7,
            transform=ccrs.PlateCarree(), zorder=5)
    text = f"{st} {NAMES.get(st, '')}"
    if st in counts:
        c = counts[st]
        text += f"\n{c['fog']} fog / {c['total'] - c['fog']} no fog"
    dx, dy, ha = LABEL.get(st, (0.35, 0.15, "left"))
    ax.text(p["lon"] + dx, p["lat"] + dy, text, fontsize=9, ha=ha, va="center",
            transform=ccrs.PlateCarree(), path_effects=halo, zorder=6)

legend = [Line2D([], [], marker="*", color="w", markerfacecolor=TEST_COLOR, markeredgecolor="black",
                 markersize=14, label=f"test airport (2023–2024)"),
          Line2D([], [], marker="o", color="w", markerfacecolor=TRAIN_COLOR, markeredgecolor="black",
                 markersize=9, label="training airports (2018–2022)")]
ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.9)

out = Path("docs/figures/stations.png")
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)