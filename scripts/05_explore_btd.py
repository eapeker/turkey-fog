from glob import glob
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from satpy import Scene

scn = Scene(reader="seviri_l1b_native", filenames=glob("data/raw/seviri/*.nat"))
scn.load(["IR_039", "IR_108"])
local = scn.crop(ll_bbox=(25.5, 35.5, 45.0, 42.5))
diff = local["IR_108"] - local["IR_039"]

print(diff.shape, float(diff.min()), float(diff.max()))

crs = local["IR_108"].attrs["area"].to_cartopy_crs()
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_extent([25.5, 45.0, 35.5, 42.5], crs=ccrs.PlateCarree())
img = ax.imshow(diff, transform=crs, extent=crs.bounds, origin="upper", cmap="RdBu_r", vmin=-2, vmax=10)
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
plt.colorbar(img, label="IR10.8 − IR3.9 (K)", shrink=0.6)
plt.title("2019-12-17 00:12 UTC")
ax.plot(32.995, 40.128, marker="^", color="black", markersize=7, transform=ccrs.PlateCarree())
plt.savefig("docs/figures/btd_20191217_0012.png", dpi=150, bbox_inches="tight")

area = local["IR_108"].attrs["area"]
col, row = area.get_array_indices_from_lonlat(32.9951, 40.1281)
print("Esenboğa BTD:", float(diff[row, col]))