import pandas as pd
import os
os.chdir("C:/Users/emina/Desktop/eumetsat/turkey-fog")

df=pd.read_csv("data/raw/LTAC.csv",na_values="M")
low_vis=df.vsby<0.62
has_fg=df["wxcodes"].str.contains("FG",na=False)

print("Görüş 1 km den düşük:",low_vis.sum())
print("iki şart birden:",(low_vis&has_fg).sum())

print(df[low_vis&has_fg][["valid","vsby","wxcodes"]])
